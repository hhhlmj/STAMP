"""
TG-GEN: Twin-Guided Grid Evolution Network (ICEWS18+ in ICEWS18 dir)

This main.py keeps full functionality:
- spatial twin (locatedin=256 + gridneighbor=257)
- merge locatedin into main snapshots (optional)
- bridge entity->grid proxy edges (optional, with rel-id filter)
- safe DGL subgraph cache (LRU)
- evaluation prints + logs: MRR + Hits@1/3/10 (raw/filter) for entity & relation
- best checkpoint selected by configurable metric (default: mrr_filter)

Run example (bridge only, select by hits@1_filter, 30 epochs):
python main.py -d ICEWS18 --gpu 0 \
  --train-history-len 3 --test-history-len 3 --dilate-len 1 \
  --lr 0.001 --n-layers 2 --n-hidden 200 \
  --evaluate-every 1 --n-epochs 30 --early-stopping-patience 10 \
  --self-loop --layer-norm \
  --decoder convtranse --encoder uvrgcn \
  --weight 0.5 --discount 1 --angle 10 --task-weight 0.7 \
  --entity-prediction --relation-prediction --add-static-graph \
  --twin-period 30 --twin-loss-weight 0 \
  --bridge-entity-to-grid --bridge-rel-ids 0,1,2 \
  --best-metric hits1_filter \
  --amp --tf32 --ablation full
"""

import argparse
import itertools
import os
import sys
import json
import math
import random
from collections import defaultdict, OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Path setup (assume this file is in TG/src/)
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from rgcn import utils
from rgcn.utils import build_sub_graph
from rgcn.knowledge_graph import _read_triplets_as_list
from src.rrgcn import RecurrentRGCN
from src.hyperparameter_range import hp_range

# -----------------------------------------------------------------------------
# Subgraph cache (LRU)
# -----------------------------------------------------------------------------
subgraph_cache: "OrderedDict[int, object]" = OrderedDict()

def clear_gpu_cache():
    global subgraph_cache
    subgraph_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_subgraph_cached(triples: np.ndarray,
                        num_nodes: int,
                        num_rels: int,
                        use_cuda: bool,
                        gpu: int,
                        max_cache_size: int = 5000):
    if triples is None or len(triples) == 0:
        triples = np.empty((0, 3), dtype=np.int64)

    device_tag = f"cuda:{gpu}" if use_cuda else "cpu"
    key = hash((device_tag, triples.tobytes(), num_nodes, num_rels))

    if key in subgraph_cache:
        g = subgraph_cache.pop(key)
        subgraph_cache[key] = g
        return g

    g = build_sub_graph(num_nodes, num_rels, triples, use_cuda=use_cuda, gpu=gpu)
    subgraph_cache[key] = g

    if len(subgraph_cache) > max_cache_size:
        subgraph_cache.popitem(last=False)

    return g

# -----------------------------------------------------------------------------
# EarlyStopping + Logger
# -----------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, verbose=True):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.verbose = bool(verbose)
        self.best_score = None
        self.best_epoch = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_score, epoch):
        if self.best_score is None:
            self.best_score = val_score
            self.best_epoch = epoch
            self.counter = 0
            return

        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"✓ Validation score improved to {val_score:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"⚠ No improvement. Patience: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"🛑 Early stopping triggered at epoch {epoch}")
                    print(f"📊 Best score: {self.best_score:.6f} at epoch {self.best_epoch}")

class TrainingLogger:
    def __init__(self, log_dir='../logs', model_name='model'):
        self.log_dir = log_dir
        self.model_name = model_name
        os.makedirs(log_dir, exist_ok=True)

        self.log_file = os.path.join(log_dir, f'{model_name}_log.json')
        self.csv_file = os.path.join(log_dir, f'{model_name}_metrics.csv')

        self.history = OrderedDict([
            ('epochs', []),
            ('train_loss', []),
            ('train_loss_e', []),
            ('train_loss_r', []),
            ('train_loss_twin', []),
            ('learning_rate', []),

            ('val_mrr_raw', []),
            ('val_mrr_filter', []),
            ('val_mrr_raw_r', []),
            ('val_mrr_filter_r', []),

            ('val_hits1_raw', []),
            ('val_hits3_raw', []),
            ('val_hits10_raw', []),
            ('val_hits1_filter', []),
            ('val_hits3_filter', []),
            ('val_hits10_filter', []),

            ('val_hits1_raw_r', []),
            ('val_hits3_raw_r', []),
            ('val_hits10_raw_r', []),
            ('val_hits1_filter_r', []),
            ('val_hits3_filter_r', []),
            ('val_hits10_filter_r', []),

            ('best_metric', []),
            ('best_metric_name', []),
        ])

    def log_epoch(self, epoch, avg_loss, loss_e, loss_r, loss_twin, lr, best_metric, best_name):
        self.history['epochs'].append(int(epoch))
        self.history['train_loss'].append(float(avg_loss))
        self.history['train_loss_e'].append(float(loss_e))
        self.history['train_loss_r'].append(float(loss_r))
        self.history['train_loss_twin'].append(float(loss_twin))
        self.history['learning_rate'].append(float(lr))
        self.history['best_metric'].append(float(best_metric))
        self.history['best_metric_name'].append(str(best_name))

    def log_validation(self, epoch,
                       mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r,
                       h1_raw, h3_raw, h10_raw, h1_f, h3_f, h10_f,
                       h1_raw_r, h3_raw_r, h10_raw_r, h1_f_r, h3_f_r, h10_f_r):
        self.history['val_mrr_raw'].append(float(mrr_raw))
        self.history['val_mrr_filter'].append(float(mrr_filter))
        self.history['val_mrr_raw_r'].append(float(mrr_raw_r))
        self.history['val_mrr_filter_r'].append(float(mrr_filter_r))

        self.history['val_hits1_raw'].append(float(h1_raw))
        self.history['val_hits3_raw'].append(float(h3_raw))
        self.history['val_hits10_raw'].append(float(h10_raw))
        self.history['val_hits1_filter'].append(float(h1_f))
        self.history['val_hits3_filter'].append(float(h3_f))
        self.history['val_hits10_filter'].append(float(h10_f))

        self.history['val_hits1_raw_r'].append(float(h1_raw_r))
        self.history['val_hits3_raw_r'].append(float(h3_raw_r))
        self.history['val_hits10_raw_r'].append(float(h10_raw_r))
        self.history['val_hits1_filter_r'].append(float(h1_f_r))
        self.history['val_hits3_filter_r'].append(float(h3_f_r))
        self.history['val_hits10_filter_r'].append(float(h10_f_r))

    def save_all(self):
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        print(f"📝 JSON log saved to {self.log_file}")

        import csv
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            headers = list(self.history.keys())
            writer.writerow(headers)
            max_len = max(len(v) for v in self.history.values())
            for i in range(max_len):
                row = [self.history[h][i] if i < len(self.history[h]) else '' for h in headers]
                writer.writerow(row)
        print(f"📝 CSV log saved to {self.csv_file}")

# -----------------------------------------------------------------------------
# Pretty printing
# -----------------------------------------------------------------------------
def print_test_table(title, ent, rel):
    print("\n" + "=" * 130)
    print(title)
    print("=" * 130)
    print(f"{'':<10}{'MRR_raw':>10} {'MRR_filt':>10} {'H@1_raw':>10} {'H@3_raw':>10} {'H@10_raw':>10} "
          f"{'H@1_filt':>10} {'H@3_filt':>10} {'H@10_filt':>10}")
    print("-" * 130)
    print(f"{'Entity':<10}{ent['mrr_raw']:>10.4f} {ent['mrr_filt']:>10.4f} {ent['h1_raw']:>10.4f} {ent['h3_raw']:>10.4f} {ent['h10_raw']:>10.4f} "
          f"{ent['h1_filt']:>10.4f} {ent['h3_filt']:>10.4f} {ent['h10_filt']:>10.4f}")
    print(f"{'Rel':<10}{rel['mrr_raw']:>10.4f} {rel['mrr_filt']:>10.4f} {rel['h1_raw']:>10.4f} {rel['h3_raw']:>10.4f} {rel['h10_raw']:>10.4f} "
          f"{rel['h1_filt']:>10.4f} {rel['h3_filt']:>10.4f} {rel['h10_filt']:>10.4f}")
    print("=" * 130)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def sort_by_time(arr: np.ndarray):
    if arr is None or len(arr) == 0:
        return arr
    idx = np.argsort(arr[:, 3], kind="mergesort")
    return arr[idx]

def get_snapshot_times(data_array: np.ndarray):
    if data_array is None or len(data_array) == 0:
        return []
    times = data_array[:, 3]
    _, indices = np.unique(times, return_index=True)
    unique_times = times[np.sort(indices)]
    return unique_times.tolist()

def build_time_to_edges_dict(triples_np):
    out = {}
    if triples_np is None or len(triples_np) == 0:
        return out
    for row in triples_np:
        tt = int(row[3])
        out.setdefault(tt, []).append(row)
    for k in list(out.keys()):
        out[k] = np.asarray(out[k], dtype=triples_np.dtype)
    return out

def build_grid_adj_dict(gridneighbor_np):
    adj = defaultdict(list)
    if gridneighbor_np is None or len(gridneighbor_np) == 0:
        return adj
    for h, r, t, _ in gridneighbor_np:
        h = int(h); t = int(t)
        adj[h].append(t)
        adj[t].append(h)
    return adj

def _grid_ctx_from_memory(grid_ids, spatial_memory_grid, device, grid_adj=None, use_grid_prop=False):
    if len(grid_ids) == 0:
        return None
    h_dim = None
    for gid in grid_ids:
        if gid in spatial_memory_grid:
            h_dim = int(spatial_memory_grid[gid].shape[-1])
            break
    if h_dim is None:
        return None
    out = torch.zeros(len(grid_ids), h_dim, device=device)
    for i, gid in enumerate(grid_ids):
        if gid in spatial_memory_grid:
            out[i] = spatial_memory_grid[gid].to(device)
    if use_grid_prop and grid_adj is not None:
        out2 = torch.zeros_like(out)
        for i, gid in enumerate(grid_ids):
            neigh = grid_adj.get(int(gid), [])
            if not neigh:
                out2[i] = out[i]
                continue
            neigh_embs = [spatial_memory_grid[n].to(device) for n in neigh if n in spatial_memory_grid]
            out2[i] = torch.stack(neigh_embs, 0).mean(0) if neigh_embs else out[i]
        out = out2
    return out

def get_spatial_twin_embeddings(input_times, entityloc_by_time, spatial_memory_grid,
                               num_ents, h_dim, use_cuda, gpu,
                               grid_adj=None, use_grid_prop=False):
    device = torch.device(f"cuda:{gpu}") if (use_cuda and gpu >= 0) else torch.device("cpu")
    twin_space_list = []
    for ts in input_times:
        ts = int(ts)
        edges = entityloc_by_time.get(ts, None) if entityloc_by_time is not None else None
        twin_space = torch.zeros(num_ents, h_dim, device=device)
        if edges is None or len(edges) == 0:
            twin_space_list.append(twin_space)
            continue
        loc2grids = defaultdict(list)
        for h, r, t, _ in edges:
            loc2grids[int(h)].append(int(t))
        for loc, grids in loc2grids.items():
            ctx = _grid_ctx_from_memory(grids, spatial_memory_grid, device, grid_adj, use_grid_prop)
            if ctx is not None:
                twin_space[loc] = ctx.mean(0)
        twin_space_list.append(twin_space)
    return twin_space_list

def update_spatial_memory_grid(time_stamp, pre_emb_cpu, entityloc_by_time, spatial_memory_grid,
                               grid_adj=None, update_neighbors=False, max_size=50000):
    ts = int(time_stamp)
    edges = entityloc_by_time.get(ts, None) if entityloc_by_time is not None else None
    if edges is None or len(edges) == 0:
        return
    grids = set(int(row[2]) for row in edges)
    if update_neighbors and grid_adj is not None:
        for g in list(grids):
            grids.update(grid_adj.get(g, []))
    for gid in grids:
        if 0 <= gid < pre_emb_cpu.shape[0]:
            spatial_memory_grid[gid] = pre_emb_cpu[gid].detach().cpu()
    if len(spatial_memory_grid) > max_size:
        # prune arbitrary
        for k in list(spatial_memory_grid.keys())[: len(spatial_memory_grid) - max_size]:
            spatial_memory_grid.pop(k, None)

def get_twin_embeddings(input_times, history_memory, period, num_ents, h_dim, use_cuda, gpu):
    device = torch.device(f"cuda:{gpu}") if (use_cuda and gpu >= 0) else torch.device("cpu")
    out = []
    for t in input_times:
        twin_time = int(t) - int(period)
        if twin_time in history_memory:
            out.append(history_memory[twin_time].to(device))
        else:
            out.append(torch.zeros(num_ents, h_dim, device=device))
    return out

def build_loc2grids_from_loc_edges(loc_edges_4col):
    loc2grids = {}
    if loc_edges_4col is None or len(loc_edges_4col) == 0:
        return loc2grids
    for h, r, t, _ in loc_edges_4col:
        loc2grids.setdefault(int(h), []).append(int(t))
    return loc2grids

def maybe_bridge_entity_to_grid_snapshot(snap_main_3col, ts, entityloc_by_time, enabled,
                                        bridge_rel_ids=None, rel_locatedin=256, per_entity_cap=16):
    if not enabled or entityloc_by_time is None or snap_main_3col is None or len(snap_main_3col) == 0:
        return None
    ts = int(ts)
    loc_edges = entityloc_by_time.get(ts, None)
    if loc_edges is None or len(loc_edges) == 0:
        return None

    if bridge_rel_ids is not None and len(bridge_rel_ids) == 0:
        bridge_rel_ids = None

    loc2grids = build_loc2grids_from_loc_edges(loc_edges)
    loc_nodes = set(loc2grids.keys())
    if not loc_nodes:
        return None

    ent2grids = {}
    for h, r, t in snap_main_3col:
        r = int(r)
        if bridge_rel_ids is not None and r not in bridge_rel_ids:
            continue
        h = int(h); t = int(t)
        if h in loc_nodes and t not in loc_nodes:
            grids = loc2grids.get(h, [])
            if grids:
                ent2grids.setdefault(t, set()).update(grids)
        elif t in loc_nodes and h not in loc_nodes:
            grids = loc2grids.get(t, [])
            if grids:
                ent2grids.setdefault(h, set()).update(grids)

    if not ent2grids:
        return None

    proxy = []
    for ent, grids in ent2grids.items():
        g_list = list(grids)
        if per_entity_cap and len(g_list) > per_entity_cap:
            g_list = g_list[:per_entity_cap]
        for g in g_list:
            proxy.append((int(ent), int(rel_locatedin), int(g)))

    return np.asarray(proxy, dtype=snap_main_3col.dtype) if proxy else None

def maybe_augment_snapshot(snap_main_3col, ts, entityloc_by_time,
                          merge_enabled, bridge_enabled,
                          bridge_rel_ids=None,
                          rel_locatedin=256):
    if snap_main_3col is None:
        return snap_main_3col
    parts = [snap_main_3col]
    proxy = maybe_bridge_entity_to_grid_snapshot(
        snap_main_3col, ts, entityloc_by_time, bridge_enabled,
        bridge_rel_ids=bridge_rel_ids, rel_locatedin=rel_locatedin
    )
    if proxy is not None and len(proxy) > 0:
        parts.append(proxy)
    if merge_enabled and entityloc_by_time is not None:
        loc_edges = entityloc_by_time.get(int(ts), None)
        if loc_edges is not None and len(loc_edges) > 0:
            parts.append(loc_edges[:, :3])  # drop time col for graph build
    if len(parts) == 1:
        return snap_main_3col
    return np.concatenate(parts, axis=0)

def _flatten_ranks(ranks_list):
    flat = []
    for r in ranks_list:
        if torch.is_tensor(r):
            flat.extend(r.view(-1).detach().cpu().numpy().tolist())
        elif isinstance(r, (list, tuple, np.ndarray)):
            flat.extend(list(r))
        else:
            flat.append(float(r))
    return np.asarray(flat, dtype=np.float64)

def calc_hits(ranks_list):
    arr = _flatten_ranks(ranks_list)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(arr <= 1)), float(np.mean(arr <= 3)), float(np.mean(arr <= 10))

def select_metric(metrics: dict, name: str):
    if name not in metrics:
        raise ValueError(f"Unknown best metric '{name}'. Available: {sorted(metrics.keys())}")
    return float(metrics[name])

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def evaluate(model,
             history_list, history_times,
             eval_list, eval_times,
             num_rels, num_nodes, use_cuda,
             all_ans_list, all_ans_r_list,
             static_graph, args,
             entityloc_by_time=None,
             grid_adj=None,
             mode="valid"):
    ranks_raw, ranks_filt = [], []
    ranks_raw_r, ranks_filt_r = [], []

    model.eval()

    # rolling window
    input_list = [snap for snap in history_list[-args.test_history_len:]]
    input_times = [t for t in history_times[-args.test_history_len:]]

    history_memory = OrderedDict()
    spatial_memory_grid = {}

    warm_margin = int(getattr(args, "twin_warmup_margin", 5))
    win_len = int(getattr(args, "train_history_len", args.test_history_len))
    warm_len = args.test_history_len + args.twin_period + warm_margin
    warm_start = max(0, len(history_list) - warm_len)
    warm_list = history_list[warm_start:]
    warm_times = history_times[warm_start:]

    def _prune(ref_time: int):
        min_keep_time = ref_time - args.twin_period - win_len - warm_margin
        while history_memory:
            oldest = next(iter(history_memory))
            if oldest < min_keep_time:
                history_memory.popitem(last=False)
            else:
                break

    bridge_rel_ids = getattr(args, "bridge_rel_ids", None)

    # warm-up fill memory
    with torch.no_grad():
        warm_win_list, warm_win_times = [], []
        for snap, ts in zip(warm_list, warm_times):
            warm_win_list.append(snap)
            warm_win_times.append(ts)
            if len(warm_win_list) > win_len:
                warm_win_list.pop(0); warm_win_times.pop(0)

            merged = [
                maybe_augment_snapshot(s, t, entityloc_by_time,
                                      args.merge_locatedin_into_main,
                                      args.bridge_entity_to_grid,
                                      bridge_rel_ids=bridge_rel_ids)
                for s, t in zip(warm_win_list, warm_win_times)
            ]
            glist = [get_subgraph_cached(g, num_nodes, num_rels, use_cuda, args.gpu,
                                         max_cache_size=args.subgraph_cache_size)
                     for g in merged]

            twin_time_h_list = None if args.ablation == "no_twin" else get_twin_embeddings(
                warm_win_times, history_memory, args.twin_period, num_nodes, args.n_hidden, use_cuda, args.gpu
            )

            twin_space_h_list = None
            if args.use_spatial_twin and entityloc_by_time is not None:
                twin_space_h_list = get_spatial_twin_embeddings(
                    warm_win_times, entityloc_by_time, spatial_memory_grid, num_nodes, args.n_hidden,
                    use_cuda, args.gpu, grid_adj=grid_adj, use_grid_prop=args.use_grid_prop
                )

            evolve_embs, _, _, _, _ = model(glist, static_graph, use_cuda,
                                            twin_time_h_list=twin_time_h_list,
                                            twin_space_h_list=twin_space_h_list)
            pre_emb = F.normalize(evolve_embs[-1]) if args.layer_norm else evolve_embs[-1]
            history_memory[int(ts)] = pre_emb.detach().cpu()
            history_memory.move_to_end(int(ts))
            _prune(int(ts))

    for time_idx, snap in enumerate(tqdm(eval_list, desc=f"Eval({mode})")):
        cur_time = int(eval_times[time_idx])

        merged = [
            maybe_augment_snapshot(s, t, entityloc_by_time,
                                  args.merge_locatedin_into_main,
                                  args.bridge_entity_to_grid,
                                  bridge_rel_ids=bridge_rel_ids)
            for s, t in zip(input_list, input_times)
        ]
        glist = [get_subgraph_cached(g, num_nodes, num_rels, use_cuda, args.gpu,
                                     max_cache_size=args.subgraph_cache_size)
                 for g in merged]

        twin_time_h_list = None if args.ablation == "no_twin" else get_twin_embeddings(
            input_times, history_memory, args.twin_period, num_nodes, args.n_hidden, use_cuda, args.gpu
        )

        twin_space_h_list = None
        if args.use_spatial_twin and entityloc_by_time is not None:
            twin_space_h_list = get_spatial_twin_embeddings(
                input_times, entityloc_by_time, spatial_memory_grid, num_nodes, args.n_hidden,
                use_cuda, args.gpu, grid_adj=grid_adj, use_grid_prop=args.use_grid_prop
            )

        tri = torch.as_tensor(snap, dtype=torch.long)
        if use_cuda and args.gpu >= 0:
            tri = tri.to(args.gpu)

        test_triples, score_ent, score_rel, pre_emb = model.predict(
            glist, num_rels, static_graph, tri, use_cuda,
            twin_time_h_list=twin_time_h_list,
            twin_space_h_list=twin_space_h_list
        )

        # update memory with last history time
        if input_times:
            last_t = int(input_times[-1])
            history_memory[last_t] = pre_emb.detach().cpu()
            history_memory.move_to_end(last_t)
            if args.use_spatial_twin and entityloc_by_time is not None:
                update_spatial_memory_grid(
                    last_t, pre_emb.detach().cpu(), entityloc_by_time, spatial_memory_grid,
                    grid_adj=grid_adj, update_neighbors=args.spatial_update_neighbors,
                    max_size=args.spatial_memory_size
                )
            _prune(cur_time)

        _, _, rr_raw_r, rr_filt_r = utils.get_total_rank(
            test_triples, score_rel, all_ans_r_list[time_idx], eval_bz=1000, rel_predict=1
        )
        _, _, rr_raw, rr_filt = utils.get_total_rank(
            test_triples, score_ent, all_ans_list[time_idx], eval_bz=1000, rel_predict=0
        )

        ranks_raw.append(rr_raw); ranks_filt.append(rr_filt)
        ranks_raw_r.append(rr_raw_r); ranks_filt_r.append(rr_filt_r)

        # roll
        input_list.pop(0); input_list.append(snap)
        input_times.pop(0); input_times.append(cur_time)

    # aggregate
    mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filt = utils.stat_ranks(ranks_filt, "filter_ent")
    mrr_raw_r = utils.stat_ranks(ranks_raw_r, "raw_rel")
    mrr_filt_r = utils.stat_ranks(ranks_filt_r, "filter_rel")

    h1_raw, h3_raw, h10_raw = calc_hits(ranks_raw)
    h1_f, h3_f, h10_f = calc_hits(ranks_filt)
    h1_raw_r, h3_raw_r, h10_raw_r = calc_hits(ranks_raw_r)
    h1_f_r, h3_f_r, h10_f_r = calc_hits(ranks_filt_r)

    ent = dict(mrr_raw=mrr_raw, mrr_filt=mrr_filt, h1_raw=h1_raw, h3_raw=h3_raw, h10_raw=h10_raw,
               h1_filt=h1_f, h3_filt=h3_f, h10_filt=h10_f)
    rel = dict(mrr_raw=mrr_raw_r, mrr_filt=mrr_filt_r, h1_raw=h1_raw_r, h3_raw=h3_raw_r, h10_raw=h10_raw_r,
               h1_filt=h1_f_r, h3_filt=h3_f_r, h10_filt=h10_f_r)
    return ent, rel

# -----------------------------------------------------------------------------
# Experiment
# -----------------------------------------------------------------------------
def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    if n_hidden is not None: args.n_hidden = int(n_hidden)
    if n_layers is not None: args.n_layers = int(n_layers)
    if dropout is not None: args.dropout = float(dropout)
    if n_bases is not None: args.n_bases = int(n_bases)

    clear_gpu_cache()

    print("loading graph data")
    data = utils.load_data(args.dataset)

    # sort
    data.train = sort_by_time(data.train)
    data.valid = sort_by_time(data.valid)
    data.test = sort_by_time(data.test)

    gridneighbor = getattr(data, "gridneighbor", None)
    entityloc_neighbor = getattr(data, "entityloc_neighbor", None)

    if gridneighbor is None:
        gridneighbor = np.empty((0, 4), dtype=np.int64)
    else:
        gridneighbor = sort_by_time(gridneighbor)

    if entityloc_neighbor is None:
        entityloc_neighbor = np.empty((0, 4), dtype=np.int64)
    else:
        entityloc_neighbor = sort_by_time(entityloc_neighbor)

    num_nodes = int(data.num_nodes)
    num_rels = int(data.num_rels)

    entityloc_by_time = build_time_to_edges_dict(entityloc_neighbor)
    grid_adj = build_grid_adj_dict(gridneighbor)

    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)

    train_times = get_snapshot_times(data.train)
    valid_times = get_snapshot_times(data.valid)
    test_times = get_snapshot_times(data.test)

    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)
    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    os.makedirs('../models', exist_ok=True)
    os.makedirs('../logs', exist_ok=True)

    # include key switches in name
    name_extra = []
    if args.use_spatial_twin: name_extra.append("spTwin")
    if args.use_grid_prop: name_extra.append("gridProp")
    if args.merge_locatedin_into_main: name_extra.append("mergeLoc")
    if args.bridge_entity_to_grid: name_extra.append("bridgeEntGrid")
    if args.bridge_rel_ids is not None: name_extra.append("bridgeRel")
    name_extra = "-".join(name_extra) if name_extra else "base"

    model_name = "{}-{}-{}-ly{}-his{}-weight{}-discount{}-angle{}-dp{}{}{}{}-gpu{}-twin{}-abl{}-{}" \
        .format(args.dataset, args.encoder, args.decoder, args.n_layers,
                args.train_history_len, args.weight, args.discount, args.angle,
                args.dropout, args.input_dropout, args.hidden_dropout, args.feat_dropout,
                args.gpu, args.twin_period, args.ablation, name_extra)
    model_state_file = os.path.join('..', 'models', model_name + ".pth")
    print("Sanity Check: model file :", model_state_file)

    # static graph
    static_graph = None
    num_static_rels, num_words = 0, 0
    if args.add_static_graph:
        static_path = os.path.join("..", "data", args.dataset, "e-w-graph.txt")
        static_triples = np.array(_read_triplets_as_list(static_path, {}, {}, load_time=False))
        if len(static_triples) > 0:
            num_static_rels = len(np.unique(static_triples[:, 1]))
            num_words = len(np.unique(static_triples[:, 2]))
            static_triples[:, 2] = static_triples[:, 2] + num_nodes
            static_node_id = torch.from_numpy(np.arange(num_words + num_nodes)).view(-1, 1).long()
            if use_cuda: static_node_id = static_node_id.to(args.gpu)
            static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)

    # model
    model = RecurrentRGCN(
        args.decoder, args.encoder,
        num_nodes, num_rels,
        num_static_rels, num_words,
        args.n_hidden, args.opn,
        sequence_len=args.train_history_len,
        num_bases=args.n_bases,
        num_basis=args.n_basis,
        num_hidden_layers=args.n_layers,
        dropout=args.dropout,
        self_loop=args.self_loop,
        skip_connect=args.skip_connect,
        layer_norm=args.layer_norm,
        input_dropout=args.input_dropout,
        hidden_dropout=args.hidden_dropout,
        feat_dropout=args.feat_dropout,
        aggregation=args.aggregation,
        weight=args.weight,
        discount=args.discount,
        angle=args.angle,
        use_static=args.add_static_graph,
        entity_prediction=args.entity_prediction,
        relation_prediction=args.relation_prediction,
        use_cuda=use_cuda,
        gpu=args.gpu,
        analysis=args.run_analysis,
        ablation=args.ablation,
        label_smoothing=args.label_smoothing
    )
    if use_cuda:
        model = model.to(args.gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = GradScaler(enabled=(use_cuda and args.amp))

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, verbose=True)
    logger = TrainingLogger(log_dir='../logs', model_name=model_name)

    # parse bridge_rel_ids (comma-separated ints)
    if args.bridge_rel_ids is not None:
        if isinstance(args.bridge_rel_ids, str) and args.bridge_rel_ids.strip():
            try:
                args.bridge_rel_ids = set(int(x) for x in args.bridge_rel_ids.split(",") if x.strip() != "")
            except Exception:
                raise ValueError("--bridge-rel-ids must be comma-separated integers, e.g. 0,12,19")
        else:
            args.bridge_rel_ids = None

    # best-metric init
    best_metric_name = args.best_metric
    best_metric_value = -1e9
    best_epoch = -1

    # memory for twin MSE (optional)
    history_memory = OrderedDict()
    mem_margin = int(getattr(args, "twin_warmup_margin", 5))
    mem_keep = args.train_history_len + args.twin_period + mem_margin

    def _mem_put(time_id, emb_tensor):
        history_memory[int(time_id)] = emb_tensor.detach().cpu()
        history_memory.move_to_end(int(time_id))
        while len(history_memory) > mem_keep:
            history_memory.popitem(last=False)

    # TRAIN
    print("---------------------------------------- start training ----------------------------------------\n")
    for epoch in range(args.n_epochs):
        model.train()
        losses, losses_e, losses_r, losses_twin = [], [], [], []

        idx = list(range(len(train_list)))
        random.shuffle(idx)

        for train_idx in tqdm(idx, desc=f"Main Task (epoch {epoch})"):
            if train_idx == 0:
                continue

            out_snap = train_list[train_idx]
            out_tri = torch.from_numpy(out_snap).long()
            if use_cuda:
                out_tri = out_tri.to(args.gpu)

            if train_idx - args.train_history_len < 0:
                hist_list = train_list[0:train_idx]
                hist_times = train_times[0:train_idx]
            else:
                hist_list = train_list[train_idx - args.train_history_len:train_idx]
                hist_times = train_times[train_idx - args.train_history_len:train_idx]

            merged_hist = [
                maybe_augment_snapshot(s, t, entityloc_by_time,
                                      args.merge_locatedin_into_main,
                                      args.bridge_entity_to_grid,
                                      bridge_rel_ids=args.bridge_rel_ids)
                for s, t in zip(hist_list, hist_times)
            ]
            glist = [get_subgraph_cached(s, num_nodes, num_rels, use_cuda, args.gpu,
                                         max_cache_size=args.subgraph_cache_size)
                     for s in merged_hist]

            twin_time_h_list = None
            if args.ablation != "no_twin":
                twin_time_h_list = get_twin_embeddings(hist_times, history_memory, args.twin_period,
                                                       num_nodes, args.n_hidden, use_cuda, args.gpu)

            twin_space_h_list = None
            if args.use_spatial_twin and entityloc_by_time is not None:
                twin_space_h_list = get_spatial_twin_embeddings(
                    hist_times, entityloc_by_time, {},  # spatial mem is built in eval; training uses zeros unless you extend it
                    num_nodes, args.n_hidden, use_cuda, args.gpu,
                    grid_adj=grid_adj, use_grid_prop=args.use_grid_prop
                )

            # build all_triples with inverse
            triples = out_tri
            inv = triples[:, [2, 1, 0]]
            inv[:, 1] = inv[:, 1] + num_rels
            all_triples = torch.cat([triples, inv], 0)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                evolve_embs, _, r_emb, _, _ = model(glist, static_graph, use_cuda,
                                                    twin_time_h_list=twin_time_h_list,
                                                    twin_space_h_list=twin_space_h_list)
                pre_emb = F.normalize(evolve_embs[-1]) if args.layer_norm else evolve_embs[-1]

                if hist_times:
                    _mem_put(hist_times[-1], pre_emb)

                loss_ent = torch.zeros((), device=pre_emb.device)
                loss_rel = torch.zeros((), device=pre_emb.device)
                loss_twin = torch.zeros((), device=pre_emb.device)

                if args.entity_prediction:
                    scores_ob = model.decoder_ob.forward(pre_emb, r_emb, all_triples, mode="train").view(-1, num_nodes)
                    loss_ent = model.loss_e(scores_ob, all_triples[:, 2])

                if args.relation_prediction and args.ablation != "no_rel_pred":
                    score_rel = model.rdecoder.forward(pre_emb, r_emb, all_triples, mode="train").view(-1, 2 * num_rels)
                    loss_rel = model.loss_r(score_rel, all_triples[:, 1])

                if args.twin_loss_weight > 0 and twin_time_h_list is not None and len(twin_time_h_list) > 0:
                    loss_twin = args.twin_loss_weight * F.mse_loss(pre_emb, twin_time_h_list[-1].to(pre_emb.device))

                if args.ablation == "no_rel_pred":
                    loss = loss_ent + loss_twin
                else:
                    loss = args.task_weight * loss_ent + (1 - args.task_weight) * loss_rel + loss_twin

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                optimizer.step()

            losses.append(float(loss.detach().cpu()))
            losses_e.append(float(loss_ent.detach().cpu()))
            losses_r.append(float(loss_rel.detach().cpu()))
            losses_twin.append(float(loss_twin.detach().cpu()))

        # VALID
        if (epoch % args.evaluate_every) == 0:
            ent_v, rel_v = evaluate(
                model,
                history_list=train_list, history_times=train_times,
                eval_list=valid_list, eval_times=valid_times,
                num_rels=num_rels, num_nodes=num_nodes, use_cuda=use_cuda,
                all_ans_list=all_ans_list_valid, all_ans_r_list=all_ans_list_r_valid,
                static_graph=static_graph, args=args,
                entityloc_by_time=entityloc_by_time, grid_adj=grid_adj,
                mode="valid"
            )
            print_test_table("VALID", ent_v, rel_v)

            metrics = {
                "mrr_raw": ent_v["mrr_raw"],
                "mrr_filter": ent_v["mrr_filt"],
                "hits1_raw": ent_v["h1_raw"],
                "hits3_raw": ent_v["h3_raw"],
                "hits10_raw": ent_v["h10_raw"],
                "hits1_filter": ent_v["h1_filt"],
                "hits3_filter": ent_v["h3_filt"],
                "hits10_filter": ent_v["h10_filt"],

                "mrr_raw_r": rel_v["mrr_raw"],
                "mrr_filter_r": rel_v["mrr_filt"],
                "hits1_raw_r": rel_v["h1_raw"],
                "hits3_raw_r": rel_v["h3_raw"],
                "hits10_raw_r": rel_v["h10_raw"],
                "hits1_filter_r": rel_v["h1_filt"],
                "hits3_filter_r": rel_v["h3_filt"],
                "hits10_filter_r": rel_v["h10_filt"],
            }
            current_metric = select_metric(metrics, best_metric_name)
            scheduler.step(current_metric)
            early_stopping(current_metric, epoch)

            if current_metric > best_metric_value:
                best_metric_value = current_metric
                best_epoch = epoch
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                print(f"\n✓ Model saved at epoch {epoch} | best {best_metric_name}={best_metric_value:.6f}\n")

            lr_now = optimizer.param_groups[0]['lr']
            logger.log_epoch(epoch, float(np.mean(losses)), float(np.mean(losses_e)), float(np.mean(losses_r)),
                             float(np.mean(losses_twin)), float(lr_now),
                             float(best_metric_value), best_metric_name)
            logger.log_validation(epoch,
                                  ent_v["mrr_raw"], ent_v["mrr_filt"], rel_v["mrr_raw"], rel_v["mrr_filt"],
                                  ent_v["h1_raw"], ent_v["h3_raw"], ent_v["h10_raw"],
                                  ent_v["h1_filt"], ent_v["h3_filt"], ent_v["h10_filt"],
                                  rel_v["h1_raw"], rel_v["h3_raw"], rel_v["h10_raw"],
                                  rel_v["h1_filt"], rel_v["h3_filt"], rel_v["h10_filt"])

            if early_stopping.early_stop:
                print(f"\n🛑 Training stopped early at epoch {epoch}")
                break

    # TEST (load best)
    if not os.path.exists(model_state_file):
        raise FileNotFoundError(f"Best checkpoint not found: {model_state_file}")
    ckpt = torch.load(model_state_file, map_location=torch.device(f"cuda:{args.gpu}") if use_cuda else torch.device("cpu"))
    model.load_state_dict(ckpt["state_dict"])
    print(f"\n📦 Loaded best checkpoint epoch={ckpt.get('epoch','?')} from {model_state_file}")

    ent_t, rel_t = evaluate(
        model,
        history_list=(train_list + valid_list), history_times=(train_times + valid_times),
        eval_list=test_list, eval_times=test_times,
        num_rels=num_rels, num_nodes=num_nodes, use_cuda=use_cuda,
        all_ans_list=all_ans_list_test, all_ans_r_list=all_ans_list_r_test,
        static_graph=static_graph, args=args,
        entityloc_by_time=entityloc_by_time, grid_adj=grid_adj,
        mode="test"
    )
    print_test_table("TEST RESULTS (Best Checkpoint)", ent_t, rel_t)

    logger.save_all()
    return ent_t, rel_t, model_state_file

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_arg_parser():
    p = argparse.ArgumentParser(description='TG-GEN: GN-GCN + Twin + Spatial(ICEWS18+)')

    p.add_argument("--ablation", type=str, default="full",
                   choices=["full", "no_twin", "no_timegate", "no_static", "no_neighbor", "no_rel_pred"])
    p.add_argument("--gpu", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("-d", "--dataset", type=str, required=True)
    p.add_argument("--test", action="store_true", default=False)
    p.add_argument("--run-analysis", action="store_true", default=False)

    p.add_argument("--train-history-len", type=int, default=3)
    p.add_argument("--test-history-len", type=int, default=3)
    p.add_argument("--dilate-len", type=int, default=1)

    p.add_argument("--decoder", type=str, default="convtranse")
    p.add_argument("--encoder", type=str, default="uvrgcn")
    p.add_argument("--opn", type=str, default="sub")

    p.add_argument("--n-bases", type=int, default=100)
    p.add_argument("--n-basis", type=int, default=100)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-hidden", type=int, default=200)

    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--input-dropout", type=float, default=0.2)
    p.add_argument("--hidden-dropout", type=float, default=0.2)
    p.add_argument("--feat-dropout", type=float, default=0.2)
    p.add_argument("--aggregation", type=str, default="none")

    p.add_argument("--self-loop", action="store_true", default=False)
    p.add_argument("--skip-connect", action="store_true", default=False)
    p.add_argument("--layer-norm", action="store_true", default=False)

    p.add_argument("--add-static-graph", action="store_true", default=False)
    p.add_argument("--weight", type=float, default=1.0)
    p.add_argument("--discount", type=float, default=1.0)
    p.add_argument("--angle", type=int, default=10)

    p.add_argument("--entity-prediction", action="store_true", default=False)
    p.add_argument("--relation-prediction", action="store_true", default=False)
    p.add_argument("--task-weight", type=float, default=0.7)

    p.add_argument("--twin-period", type=int, default=12)
    p.add_argument("--twin-warmup-margin", type=int, default=5)
    p.add_argument("--twin-loss-weight", type=float, default=0.0)

    p.add_argument("--use-spatial-twin", action="store_true", default=False)
    p.add_argument("--use-grid-prop", action="store_true", default=False)
    p.add_argument("--spatial-memory-size", type=int, default=50000)
    p.add_argument("--spatial-update-neighbors", action="store_true", default=False)

    p.add_argument("--merge-locatedin-into-main", action="store_true", default=False)
    p.add_argument("--bridge-entity-to-grid", action="store_true", default=False)
    p.add_argument("--bridge-rel-ids", type=str, default=None,
                   help="Comma-separated relation ids that are allowed to trigger bridging, e.g. 0,12,45. "
                        "If omitted, all relations trigger bridging (more noise).")

    p.add_argument("--label-smoothing", type=float, default=0.05)

    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--early-stopping-patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-norm", type=float, default=1.0)
    p.add_argument("--evaluate-every", type=int, default=1)

    p.add_argument("--amp", action="store_true", default=False)
    p.add_argument("--tf32", action="store_true", default=False)

    p.add_argument("--subgraph-cache-size", type=int, default=5000)

    p.add_argument("--best-metric", type=str, default="mrr_filter",
                   choices=[
                       "mrr_raw","mrr_filter","hits1_raw","hits3_raw","hits10_raw","hits1_filter","hits3_filter","hits10_filter",
                       "mrr_raw_r","mrr_filter_r","hits1_raw_r","hits3_raw_r","hits10_raw_r","hits1_filter_r","hits3_filter_r","hits10_filter_r"
                   ],
                   help="Metric used to save best checkpoint and drive early stopping / LR scheduler.")

    # grid search (kept for compatibility)
    p.add_argument("--grid-search", action="store_true", default=False)
    p.add_argument("-tune", "--tune", type=str, default="n_hidden,n_layers,dropout,n_bases")

    return p

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    print(args)

    if args.grid_search:
        out_log = f'{args.dataset}.{args.encoder}-{args.decoder}.gs'
        with open(out_log, 'w') as o_f:
            o_f.write("** Grid Search **\n")
        hyperparameters = args.tune.split(',') if args.tune else []
        if not hyperparameters:
            print("No hyperparameter specified.")
            sys.exit(0)

        grid = hp_range[hyperparameters[0]]
        for hp in hyperparameters[1:]:
            grid = itertools.product(grid, hp_range[hp])
        grid = list(grid)

        for i, grid_entry in enumerate(list(grid)):
            if not isinstance(grid_entry, (list, tuple)):
                grid_entry = [grid_entry]
            grid_entry = utils.flatten(grid_entry)

            m_ent, m_rel, ckpt = run_experiment(args, grid_entry[0], grid_entry[1], grid_entry[2], grid_entry[3])
            with open(out_log, 'a') as o_f:
                o_f.write(f"set {i}: {grid_entry}\n")
                o_f.write(f"best_ckpt: {ckpt}\n")
    else:
        run_experiment(args)