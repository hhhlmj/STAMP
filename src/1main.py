"""
TG-GEN: Twin-Guided Grid Evolution Network

- Safe split by time + twin memory
- Spatial twin (locatedin + GeoSOT grid) support
- Optional merge locatedin into main snapshots
- Optional bridge entity to grid proxy edges
- Full metrics: MRR + Hits@1/3/10 (raw & filter) for entity & relation
- Best checkpoint selectable via --best-metric
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

# ---------------------------------------------------------------------
# Path setup (assume this file is in src/)
# ---------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from rgcn import utils
from rgcn.utils import build_sub_graph
from rgcn.knowledge_graph import _read_triplets_as_list
from src.rrgcn import RecurrentRGCN
from src.hyperparameter_range import hp_range

# ---------------------------------------------------------------------
# Subgraph cache (LRU)
# ---------------------------------------------------------------------
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
    """
    Build/cached DGL subgraph with correct device.
    Cache key includes device tag to avoid CPU/GPU mixing.
    """
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

# ---------------------------------------------------------------------
# EarlyStopping + Logger
# ---------------------------------------------------------------------
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
                print(f"✓ Validation metric improved to {val_score:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"⚠ No improvement. Patience: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"🛑 Early stopping triggered at epoch {epoch}")
                    print(f"📊 Best metric: {self.best_score:.6f} at epoch {self.best_epoch}")

class TrainingLogger:
    def __init__(self, log_dir='../logs', model_name='model'):
        self.log_dir = log_dir
        self.model_name = model_name
        os.makedirs(log_dir, exist_ok=True)

        self.log_file = os.path.join(log_dir, f'{model_name}_log.json')
        self.csv_file = os.path.join(log_dir, f'{model_name}_metrics.csv')

        self.history = {
            'epochs': [],
            'train_loss': [],
            'train_loss_e': [],
            'train_loss_r': [],
            'train_loss_twin': [],
            'learning_rate': [],

            'val_mrr_raw': [],
            'val_mrr_filter': [],
            'val_mrr_raw_r': [],
            'val_mrr_filter_r': [],

            'val_hits1_raw': [],
            'val_hits3_raw': [],
            'val_hits10_raw': [],
            'val_hits1_filter': [],
            'val_hits3_filter': [],
            'val_hits10_filter': [],

            'val_hits1_raw_r': [],
            'val_hits3_raw_r': [],
            'val_hits10_raw_r': [],
            'val_hits1_filter_r': [],
            'val_hits3_filter_r': [],
            'val_hits10_filter_r': [],

            'best_metric': [],
            'best_metric_name': []
        }

    def log_epoch(self, epoch, avg_loss, loss_e, loss_r, loss_twin, lr, best_metric, best_metric_name):
        self.history['epochs'].append(int(epoch))
        self.history['train_loss'].append(float(avg_loss))
        self.history['train_loss_e'].append(float(loss_e))
        self.history['train_loss_r'].append(float(loss_r))
        self.history['train_loss_twin'].append(float(loss_twin))
        self.history['learning_rate'].append(float(lr))
        self.history['best_metric'].append(float(best_metric))
        self.history['best_metric_name'].append(str(best_metric_name))

    def log_validation(self, metrics: dict):
        # entity
        self.history['val_mrr_raw'].append(float(metrics['mrr_raw']))
        self.history['val_mrr_filter'].append(float(metrics['mrr_filter']))
        self.history['val_hits1_raw'].append(float(metrics['hits_raw'][1]))
        self.history['val_hits3_raw'].append(float(metrics['hits_raw'][3]))
        self.history['val_hits10_raw'].append(float(metrics['hits_raw'][10]))
        self.history['val_hits1_filter'].append(float(metrics['hits_filter'][1]))
        self.history['val_hits3_filter'].append(float(metrics['hits_filter'][3]))
        self.history['val_hits10_filter'].append(float(metrics['hits_filter'][10]))

        # relation
        self.history['val_mrr_raw_r'].append(float(metrics['mrr_raw_r']))
        self.history['val_mrr_filter_r'].append(float(metrics['mrr_filter_r']))
        self.history['val_hits1_raw_r'].append(float(metrics['hits_raw_r'][1]))
        self.history['val_hits3_raw_r'].append(float(metrics['hits_raw_r'][3]))
        self.history['val_hits10_raw_r'].append(float(metrics['hits_raw_r'][10]))
        self.history['val_hits1_filter_r'].append(float(metrics['hits_filter_r'][1]))
        self.history['val_hits3_filter_r'].append(float(metrics['hits_filter_r'][3]))
        self.history['val_hits10_filter_r'].append(float(metrics['hits_filter_r'][10]))

    def save_json(self):
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        print(f"📝 JSON log saved to {self.log_file}")

    def save_csv(self):
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

    def save_all(self):
        self.save_json()
        self.save_csv()

# ---------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------
def print_epoch_header(n_epochs):
    print("\n" + "=" * 120)
    print(f"{'Epoch':<8} {'Progress':<12} {'AvgLoss':<10} {'Loss_E':<10} {'Loss_R':<10} "
          f"{'Loss_Twin':<10} {'LR':<10} {'BestMetric':<12}")
    print("=" * 120)

def print_epoch_info(epoch, n_epochs, avg_loss, loss_e, loss_r, loss_twin, lr, best_metric):
    progress = f"{epoch+1}/{n_epochs}"
    pct = f"({(epoch+1)/n_epochs*100:.1f}%)"
    print(f"{progress:<8} {pct:<12} {avg_loss:<10.4f} {loss_e:<10.4f} {loss_r:<10.4f} "
          f"{loss_twin:<10.6f} {lr:<10.6f} {best_metric:<12.6f}")

def print_valid_table(metrics: dict):
    print("\nValid      MRR_raw    MRR_filt   H@1_raw    H@3_raw    H@10_raw   H@1_filt   H@3_filt   H@10_filt")
    print("-" * 130)
    print(f"Entity     {metrics['mrr_raw']:<9.4f} {metrics['mrr_filter']:<9.4f} "
          f"{metrics['hits_raw'][1]:<9.4f} {metrics['hits_raw'][3]:<9.4f} {metrics['hits_raw'][10]:<9.4f} "
          f"{metrics['hits_filter'][1]:<9.4f} {metrics['hits_filter'][3]:<9.4f} {metrics['hits_filter'][10]:<9.4f}")
    print(f"Rel        {metrics['mrr_raw_r']:<9.4f} {metrics['mrr_filter_r']:<9.4f} "
          f"{metrics['hits_raw_r'][1]:<9.4f} {metrics['hits_raw_r'][3]:<9.4f} {metrics['hits_raw_r'][10]:<9.4f} "
          f"{metrics['hits_filter_r'][1]:<9.4f} {metrics['hits_filter_r'][3]:<9.4f} {metrics['hits_filter_r'][10]:<9.4f}")

def print_test_table(metrics: dict):
    print("\n" + "=" * 120)
    print("✅ EVALUATION COMPLETE - Test Results (Entity & Relation)")
    print("=" * 120)
    print_valid_table(metrics)
    print("=" * 120 + "\n")

# ---------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------
def sort_by_time(arr: np.ndarray):
    if arr is None or len(arr) == 0:
        return arr
    idx = np.argsort(arr[:, 3], kind="mergesort")
    return arr[idx]

def filter_neighbor_triples(triples: np.ndarray, num_nodes: int):
    if triples is None or len(triples) == 0:
        return triples
    valid_mask = (triples[:, 0] < num_nodes) & (triples[:, 2] < num_nodes) & (triples[:, 0] >= 0) & (triples[:, 2] >= 0)
    filtered = triples[valid_mask]
    removed = len(triples) - len(filtered)
    if removed > 0:
        print(f"[Warning] Filtered {removed} invalid neighbor triples (valid entity range: 0-{num_nodes-1})")
    if len(filtered) == 0:
        return np.empty((0, triples.shape[1]), dtype=triples.dtype)
    return filtered

def get_snapshot_times(data_array: np.ndarray):
    if data_array is None or len(data_array) == 0:
        return []
    times = data_array[:, 3]
    _, indices = np.unique(times, return_index=True)
    unique_times = times[np.sort(indices)]
    return unique_times.tolist()

def get_twin_embeddings(input_list, input_times, history_memory, period,
                        num_ents, h_dim, use_cuda, gpu):
    device = torch.device(f"cuda:{gpu}") if use_cuda else torch.device("cpu")
    out = []
    for i in range(len(input_list)):
        current_time = int(input_times[i]) if i < len(input_times) else 0
        twin_time = current_time - period
        if twin_time in history_memory:
            out.append(history_memory[twin_time].to(device))
        else:
            out.append(torch.zeros(num_ents, h_dim, device=device))
    return out

# ---------------------------------------------------------------------
# Spatial helpers (ICEWS18+ inside ICEWS18 dir)
# ---------------------------------------------------------------------
def build_time_to_edges_dict(triples_np):
    time_dict = {}
    if triples_np is None or len(triples_np) == 0:
        return time_dict
    for row in triples_np:
        tt = int(row[3])
        time_dict.setdefault(tt, []).append(row)
    for k in list(time_dict.keys()):
        time_dict[k] = np.asarray(time_dict[k], dtype=triples_np.dtype)
    return time_dict

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
    embs = []
    for gid in grid_ids:
        if gid in spatial_memory_grid:
            embs.append(spatial_memory_grid[gid].to(device))
        else:
            embs.append(None)

    h_dim = None
    for e in embs:
        if e is not None:
            h_dim = e.shape[-1]
            break
    if h_dim is None:
        return None

    out = torch.zeros(len(grid_ids), h_dim, device=device)
    for i, e in enumerate(embs):
        if e is not None:
            out[i] = e

    if use_grid_prop and grid_adj is not None:
        out2 = torch.zeros_like(out)
        for i, gid in enumerate(grid_ids):
            neigh = grid_adj.get(int(gid), [])
            if len(neigh) == 0:
                out2[i] = out[i]
                continue
            neigh_embs = []
            for ng in neigh:
                if ng in spatial_memory_grid:
                    neigh_embs.append(spatial_memory_grid[ng].to(device))
            if len(neigh_embs) == 0:
                out2[i] = out[i]
            else:
                out2[i] = torch.stack(neigh_embs, dim=0).mean(dim=0)
        out = out2
    return out

def get_spatial_twin_embeddings(input_times, entityloc_by_time, spatial_memory_grid,
                               num_ents, h_dim, use_cuda, gpu,
                               grid_adj=None, use_grid_prop=False):
    device = torch.device(gpu) if use_cuda else torch.device('cpu')
    twin_space_list = []
    for ts in input_times:
        ts = int(ts)
        edges = entityloc_by_time.get(ts, None)
        twin_space = torch.zeros(num_ents, h_dim, device=device)
        if edges is None or len(edges) == 0:
            twin_space_list.append(twin_space)
            continue
        loc2grids = defaultdict(list)
        for h, r, t, _ in edges:
            loc2grids[int(h)].append(int(t))
        for loc, grids in loc2grids.items():
            ctx = _grid_ctx_from_memory(grids, spatial_memory_grid, device, grid_adj, use_grid_prop)
            if ctx is None:
                continue
            twin_space[loc] = ctx.mean(dim=0)
        twin_space_list.append(twin_space)
    return twin_space_list

def update_spatial_memory_grid(time_stamp, pre_emb_cpu, entityloc_by_time, spatial_memory_grid,
                               grid_adj=None, update_neighbors=False, max_size=50000):
    ts = int(time_stamp)
    edges = entityloc_by_time.get(ts, None)
    if edges is None or len(edges) == 0:
        return
    grids = set(int(row[2]) for row in edges)
    if update_neighbors and grid_adj is not None:
        for g in list(grids):
            for ng in grid_adj.get(g, []):
                grids.add(int(ng))
    for gid in grids:
        if 0 <= gid < pre_emb_cpu.shape[0]:
            spatial_memory_grid[gid] = pre_emb_cpu[gid].detach().cpu()
    if len(spatial_memory_grid) > max_size:
        for k in list(spatial_memory_grid.keys())[: len(spatial_memory_grid) - max_size]:
            spatial_memory_grid.pop(k, None)

# ---------------------------------------------------------------------
# merge/bridge augmentation
# ---------------------------------------------------------------------
def build_loc2grids_from_loc_edges(loc_edges):
    loc2grids = {}
    if loc_edges is None or len(loc_edges) == 0:
        return loc2grids
    for h, _, t in loc_edges:
        loc2grids.setdefault(int(h), []).append(int(t))
    return loc2grids

def maybe_bridge_entity_to_grid_snapshot(snap_main, ts, entityloc_by_time, enabled,
                                        rel_locatedin=256, per_entity_cap=16):
    if not enabled or entityloc_by_time is None or snap_main is None:
        return None
    if len(snap_main) == 0:
        return None

    ts = int(ts)
    loc_edges = entityloc_by_time.get(ts, None)
    if loc_edges is None or len(loc_edges) == 0:
        return None

    loc2grids = build_loc2grids_from_loc_edges(loc_edges[:, :3])
    if len(loc2grids) == 0:
        return None
    loc_nodes = set(loc2grids.keys())

    ent2grids = {}
    for h, r, t in snap_main:
        h = int(h); t = int(t)
        if h in loc_nodes and t not in loc_nodes:
            grids = loc2grids.get(h, [])
            if grids:
                ent2grids.setdefault(t, set()).update(grids)
        elif t in loc_nodes and h not in loc_nodes:
            grids = loc2grids.get(t, [])
            if grids:
                ent2grids.setdefault(h, set()).update(grids)

    if len(ent2grids) == 0:
        return None

    proxy = []
    for ent, grids in ent2grids.items():
        g_list = list(grids)
        if per_entity_cap is not None and per_entity_cap > 0 and len(g_list) > per_entity_cap:
            g_list = g_list[:per_entity_cap]
        for g in g_list:
            proxy.append((ent, rel_locatedin, int(g)))

    if len(proxy) == 0:
        return None
    return np.asarray(proxy, dtype=snap_main.dtype)

def maybe_augment_snapshot(snap_main, ts, entityloc_by_time, merge_enabled, bridge_enabled, rel_locatedin=256):
    if snap_main is None:
        return snap_main
    parts = [snap_main]

    proxy = maybe_bridge_entity_to_grid_snapshot(
        snap_main, ts, entityloc_by_time,
        enabled=bridge_enabled,
        rel_locatedin=rel_locatedin,
        per_entity_cap=16
    )
    if proxy is not None and len(proxy) > 0:
        parts.append(proxy)

    if merge_enabled and entityloc_by_time is not None:
        loc_edges = entityloc_by_time.get(int(ts), None)
        if loc_edges is not None and len(loc_edges) > 0:
            parts.append(loc_edges[:, :3])  # use (h,r,t)

    if len(parts) == 1:
        return snap_main
    try:
        return np.concatenate(parts, axis=0)
    except Exception:
        return snap_main

# ---------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------
def _flatten_ranks(ranks_list):
    flat = []
    for r in ranks_list:
        if torch.is_tensor(r):
            flat.extend(r.view(-1).detach().cpu().numpy().tolist())
        elif isinstance(r, (list, tuple, np.ndarray)):
            flat.extend(list(r))
        else:
            flat.append(r)
    return np.asarray(flat, dtype=np.int64)

def calc_hits_from_ranks(ranks_list, ks=(1, 3, 10)):
    arr = _flatten_ranks(ranks_list)
    if arr.size == 0:
        return {k: 0.0 for k in ks}
    out = {}
    for k in ks:
        out[k] = float(np.mean(arr <= k))
    return out

def build_metric_selector(metrics: dict):
    """
    Create a flat dict that can be indexed by args.best_metric.
    """
    sel = {}

    # entity
    sel['mrr_raw'] = metrics['mrr_raw']
    sel['mrr_filter'] = metrics['mrr_filter']
    sel['hits1_raw'] = metrics['hits_raw'][1]
    sel['hits3_raw'] = metrics['hits_raw'][3]
    sel['hits10_raw'] = metrics['hits_raw'][10]
    sel['hits1_filter'] = metrics['hits_filter'][1]
    sel['hits3_filter'] = metrics['hits_filter'][3]
    sel['hits10_filter'] = metrics['hits_filter'][10]

    # relation
    sel['mrr_raw_r'] = metrics['mrr_raw_r']
    sel['mrr_filter_r'] = metrics['mrr_filter_r']
    sel['hits1_raw_r'] = metrics['hits_raw_r'][1]
    sel['hits3_raw_r'] = metrics['hits_raw_r'][3]
    sel['hits10_raw_r'] = metrics['hits_raw_r'][10]
    sel['hits1_filter_r'] = metrics['hits_filter_r'][1]
    sel['hits3_filter_r'] = metrics['hits_filter_r'][3]
    sel['hits10_filter_r'] = metrics['hits_filter_r'][10]

    return sel

# ---------------------------------------------------------------------
# Evaluation (valid/test)
# ---------------------------------------------------------------------
def test(model,
         history_list, history_times,
         test_list, test_times,
         num_rels, num_nodes, use_cuda,
         all_ans_list, all_ans_r_list,
         model_name, static_graph,
         mode, args,
         entityloc_by_time=None,
         grid_adj=None):
    """
    Return metrics dict including MRR & Hits@1/3/10 for entity and relation.
    """
    assert len(history_list) == len(history_times), "history_list/history_times length mismatch"
    assert len(test_list) == len(test_times), "test_list/test_times length mismatch"

    ranks_raw, ranks_filter = [], []
    ranks_raw_r, ranks_filter_r = [], []

    if mode == "test":
        if not os.path.exists(model_name):
            raise FileNotFoundError(f"Checkpoint not found: {model_name}")
        map_loc = torch.device(f"cuda:{args.gpu}") if (use_cuda and args.gpu >= 0) else torch.device("cpu")
        checkpoint = torch.load(model_name, map_location=map_loc)
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    input_list = [snap for snap in history_list[-args.test_history_len:]]
    input_times = [t for t in history_times[-args.test_history_len:]]

    history_memory = OrderedDict()
    spatial_memory_grid = {}

    warm_margin = getattr(args, "twin_warmup_margin", 5)
    win_len = getattr(args, "train_history_len", args.test_history_len)
    warm_len = args.test_history_len + args.twin_period + warm_margin
    warm_start = max(0, len(history_list) - warm_len)
    warm_list = history_list[warm_start:]
    warm_times = history_times[warm_start:]

    def _prune(ref_time: int):
        min_keep_time = ref_time - args.twin_period - win_len - warm_margin
        while len(history_memory) > 0:
            oldest_time = next(iter(history_memory))
            if oldest_time < min_keep_time:
                history_memory.popitem(last=False)
            else:
                break

    # warm-up memory
    with torch.no_grad():
        warm_win_list, warm_win_times = [], []
        for snap, ts in zip(warm_list, warm_times):
            warm_win_list.append(snap)
            warm_win_times.append(ts)
            if len(warm_win_list) > win_len:
                warm_win_list.pop(0)
                warm_win_times.pop(0)

            merged_warm = [
                maybe_augment_snapshot(
                    s, t, entityloc_by_time,
                    getattr(args, 'merge_locatedin_into_main', False),
                    getattr(args, 'bridge_entity_to_grid', False),
                    rel_locatedin=256
                )
                for s, t in zip(warm_win_list, warm_win_times)
            ]
            history_glist = [
                get_subgraph_cached(g, num_nodes, num_rels, use_cuda, args.gpu, max_cache_size=args.subgraph_cache_size)
                for g in merged_warm
            ]

            if getattr(args, "ablation", "") == "no_twin":
                twin_time_h_list = None
            else:
                twin_time_h_list = get_twin_embeddings(
                    warm_win_list, warm_win_times, history_memory,
                    args.twin_period, num_nodes, args.n_hidden, use_cuda, args.gpu
                )

            twin_space_h_list = None
            if getattr(args, 'use_spatial_twin', False) and entityloc_by_time is not None:
                twin_space_h_list = get_spatial_twin_embeddings(
                    warm_win_times, entityloc_by_time, spatial_memory_grid,
                    num_nodes, args.n_hidden, use_cuda, args.gpu,
                    grid_adj=grid_adj, use_grid_prop=getattr(args, 'use_grid_prop', False)
                )

            evolve_embs, _, _, _, _ = model(
                history_glist, static_graph, use_cuda,
                twin_time_h_list=twin_time_h_list,
                twin_space_h_list=twin_space_h_list
            )
            pre_emb = F.normalize(evolve_embs[-1]) if args.layer_norm else evolve_embs[-1]
            history_memory[int(ts)] = pre_emb.detach().cpu()
            history_memory.move_to_end(int(ts))
            _prune(int(ts))

    # main loop
    for time_idx, test_snap in enumerate(tqdm(test_list, desc=f"Eval({mode})")):
        cur_time = int(test_times[time_idx])

        merged_input = [
            maybe_augment_snapshot(
                s, t, entityloc_by_time,
                getattr(args, 'merge_locatedin_into_main', False),
                getattr(args, 'bridge_entity_to_grid', False),
                rel_locatedin=256
            )
            for s, t in zip(input_list, input_times)
        ]
        history_glist = [
            get_subgraph_cached(g, num_nodes, num_rels, use_cuda, args.gpu, max_cache_size=args.subgraph_cache_size)
            for g in merged_input
        ]

        if getattr(args, "ablation", "") == "no_twin":
            twin_time_h_list = None
        else:
            twin_time_h_list = get_twin_embeddings(
                input_list, input_times, history_memory,
                args.twin_period, num_nodes, args.n_hidden, use_cuda, args.gpu
            )

        twin_space_h_list = None
        if getattr(args, 'use_spatial_twin', False) and entityloc_by_time is not None:
            twin_space_h_list = get_spatial_twin_embeddings(
                input_times, entityloc_by_time, spatial_memory_grid,
                num_nodes, args.n_hidden, use_cuda, args.gpu,
                grid_adj=grid_adj, use_grid_prop=getattr(args, 'use_grid_prop', False)
            )

        test_triples_input = torch.LongTensor(test_snap)
        if use_cuda and args.gpu >= 0:
            test_triples_input = test_triples_input.to(args.gpu)

        test_triples, final_score, final_r_score, pre_emb = model.predict(
            history_glist, num_rels, static_graph, test_triples_input, use_cuda,
            twin_time_h_list=twin_time_h_list,
            twin_space_h_list=twin_space_h_list
        )

        if len(input_times) > 0:
            last_snap_time = int(input_times[-1])
            history_memory[last_snap_time] = pre_emb.detach().cpu()
            history_memory.move_to_end(last_snap_time)

            if getattr(args, 'use_spatial_twin', False) and entityloc_by_time is not None:
                update_spatial_memory_grid(
                    last_snap_time, pre_emb.detach().cpu(),
                    entityloc_by_time, spatial_memory_grid,
                    grid_adj=grid_adj,
                    update_neighbors=getattr(args, 'spatial_update_neighbors', False),
                    max_size=getattr(args, 'spatial_memory_size', 50000)
                )

            _prune(cur_time)

        _, _, rank_raw_r, rank_filter_r = utils.get_total_rank(
            test_triples, final_r_score, all_ans_r_list[time_idx], eval_bz=1000, rel_predict=1
        )
        _, _, rank_raw, rank_filter = utils.get_total_rank(
            test_triples, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0
        )

        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        ranks_raw_r.append(rank_raw_r)
        ranks_filter_r.append(rank_filter_r)

        # roll window
        if args.multi_step:
            if not args.relation_evaluation:
                predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk)
            else:
                predicted_snap = utils.construct_snap_r(test_triples, num_nodes, num_rels, final_r_score, args.topk)
            if len(predicted_snap):
                input_list.pop(0); input_list.append(predicted_snap)
                input_times.pop(0); input_times.append(cur_time)
        else:
            input_list.pop(0); input_list.append(test_snap)
            input_times.pop(0); input_times.append(cur_time)

    mrr_raw = float(utils.stat_ranks(ranks_raw, "raw_ent"))
    mrr_filter = float(utils.stat_ranks(ranks_filter, "filter_ent"))
    mrr_raw_r = float(utils.stat_ranks(ranks_raw_r, "raw_rel"))
    mrr_filter_r = float(utils.stat_ranks(ranks_filter_r, "filter_rel"))

    hits_raw = calc_hits_from_ranks(ranks_raw, ks=(1, 3, 10))
    hits_filter = calc_hits_from_ranks(ranks_filter, ks=(1, 3, 10))
    hits_raw_r = calc_hits_from_ranks(ranks_raw_r, ks=(1, 3, 10))
    hits_filter_r = calc_hits_from_ranks(ranks_filter_r, ks=(1, 3, 10))

    metrics = {
        'mrr_raw': mrr_raw,
        'mrr_filter': mrr_filter,
        'mrr_raw_r': mrr_raw_r,
        'mrr_filter_r': mrr_filter_r,
        'hits_raw': hits_raw,
        'hits_filter': hits_filter,
        'hits_raw_r': hits_raw_r,
        'hits_filter_r': hits_filter_r
    }

    if mode == "test":
        print_test_table(metrics)

    return metrics

# ---------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------
def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    if n_hidden is not None: args.n_hidden = int(n_hidden)
    if n_layers is not None: args.n_layers = int(n_layers)
    if dropout is not None: args.dropout = float(dropout)
    if n_bases is not None: args.n_bases = int(n_bases)

    clear_gpu_cache()

    print("loading graph data")
    data = utils.load_data(args.dataset)

    data.train = sort_by_time(data.train)
    data.valid = sort_by_time(data.valid)
    data.test = sort_by_time(data.test)

    gridneighbor = getattr(data, "gridneighbor", None)
    entityloc_neighbor = getattr(data, "entityloc_neighbor", None)

    if gridneighbor is not None and len(gridneighbor) > 0:
        gridneighbor = sort_by_time(gridneighbor)
    if entityloc_neighbor is not None and len(entityloc_neighbor) > 0:
        entityloc_neighbor = sort_by_time(entityloc_neighbor)

    num_nodes = int(data.num_nodes)
    num_rels = int(data.num_rels)

    if gridneighbor is None:
        gridneighbor = np.empty((0, 4), dtype=np.int64)
    if entityloc_neighbor is None:
        entityloc_neighbor = np.empty((0, 4), dtype=np.int64)

    gridneighbor = filter_neighbor_triples(gridneighbor, num_nodes)
    entityloc_neighbor = filter_neighbor_triples(entityloc_neighbor, num_nodes)

    entityloc_by_time = build_time_to_edges_dict(entityloc_neighbor)
    grid_adj = build_grid_adj_dict(gridneighbor)

    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)

    train_times = get_snapshot_times(data.train)
    valid_times = get_snapshot_times(data.valid)
    test_times = get_snapshot_times(data.test)

    entityloc_neighbor_list = utils.split_by_time(entityloc_neighbor) if len(entityloc_neighbor) else []

    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)
    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    os.makedirs('../models', exist_ok=True)
    os.makedirs('../logs', exist_ok=True)

    # IMPORTANT: checkpoint name includes best-metric
    model_name = "{}-{}-{}-ly{}-dilate{}-his{}-weight{}-discount{}-angle{}-dp{}{}{}{}-gpu{}-twin{}-abl{}-best{}" \
        .format(args.dataset, args.encoder, args.decoder, args.n_layers, args.dilate_len,
                args.train_history_len, args.weight, args.discount, args.angle,
                args.dropout, args.input_dropout, args.hidden_dropout, args.feat_dropout,
                args.gpu, args.twin_period, args.ablation, args.best_metric)

    model_state_file = '../models/{}'.format(model_name)
    print(f"Sanity Check: model file : {model_state_file}")

    # static graph
    static_graph = None
    num_static_rels, num_words = 0, 0
    if args.add_static_graph:
        static_path = os.path.join("..", "data", args.dataset, "e-w-graph.txt")
        if not os.path.exists(static_path):
            raise FileNotFoundError(f"Static graph file not found: {static_path}")
        static_triples = np.array(_read_triplets_as_list(static_path, {}, {}, load_time=False))
        if len(static_triples) > 0:
            num_static_rels = len(np.unique(static_triples[:, 1]))
            num_words = len(np.unique(static_triples[:, 2]))
            static_triples[:, 2] = static_triples[:, 2] + num_nodes
            static_node_id = torch.from_numpy(np.arange(num_words + num_nodes)).view(-1, 1).long()
            if use_cuda:
                static_node_id = static_node_id.to(args.gpu)
            static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)

    # build model (rrgcn already supports twin_time_h_list + twin_space_h_list)
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
        ablation=args.ablation
    )

    if use_cuda:
        model = model.to(args.gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        if getattr(args, "tf32", False):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    scaler = GradScaler(enabled=(use_cuda and getattr(args, "amp", False)))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    early_stopping = EarlyStopping(patience=args.early_stopping_patience, verbose=True)
    logger = TrainingLogger(log_dir='../logs', model_name=model_name)

    print("\n" + "=" * 90)
    print("🔧 MODEL CONFIGURATION")
    print("=" * 90)
    print(f"Dataset:           {args.dataset}")
    print(f"Encoder/Decoder:   {args.encoder} / {args.decoder}")
    print(f"Hidden/Layers:     {args.n_hidden} / {args.n_layers}")
    print(f"LR/Dropout:        {args.lr} / {args.dropout}")
    print(f"GPU:               {'cuda:'+str(args.gpu) if use_cuda else 'CPU'}")
    print(f"Twin Period:       {args.twin_period}")
    print(f"Spatial Twin:      {getattr(args,'use_spatial_twin', False)} | GridProp: {getattr(args,'use_grid_prop', False)}")
    print(f"Merge locatedin:   {getattr(args,'merge_locatedin_into_main', False)} | Bridge: {getattr(args,'bridge_entity_to_grid', False)}")
    print(f"Best Metric:       {args.best_metric}")
    print("=" * 90 + "\n")

    # TEST ONLY
    if args.test:
        if not os.path.exists(model_state_file):
            print(f"-------------- {model_state_file} not exist. ----------------")
            return None
        metrics = test(
            model,
            history_list=(train_list + valid_list), history_times=(train_times + valid_times),
            test_list=test_list, test_times=test_times,
            num_rels=num_rels, num_nodes=num_nodes, use_cuda=use_cuda,
            all_ans_list=all_ans_list_test, all_ans_r_list=all_ans_list_r_test,
            model_name=model_state_file, static_graph=static_graph,
            mode="test", args=args,
            entityloc_by_time=entityloc_by_time,
            grid_adj=grid_adj
        )
        return metrics

    # TRAIN
    print("---------------------------------------- start training ----------------------------------------\n")
    best_metric_val = -1.0
    best_epoch = -1
    print_epoch_header(args.n_epochs)

    history_memory = OrderedDict()
    spatial_memory_grid = {}

    mem_margin = getattr(args, "twin_warmup_margin", 5)
    mem_keep = args.train_history_len + args.twin_period + mem_margin

    def _mem_put(time_id, emb_tensor):
        history_memory[int(time_id)] = emb_tensor.detach().cpu()
        history_memory.move_to_end(int(time_id))
        while len(history_memory) > mem_keep:
            history_memory.popitem(last=False)

    for epoch in range(args.n_epochs):
        model.train()
        losses, losses_e, losses_r, losses_twin = [], [], [], []

        idx = list(range(len(train_list)))
        random.shuffle(idx)

        # Neighbor task (optional)
        do_neighbor = (args.ablation != "no_neighbor") and (len(entityloc_neighbor_list) > 0)
        neighbor_epochs = int(getattr(args, "neighbor_epochs", 0))
        if do_neighbor and neighbor_epochs > 0:
            do_neighbor = (epoch < neighbor_epochs)

        if do_neighbor:
            idx_neighbor = list(range(len(entityloc_neighbor_list)))
            random.shuffle(idx_neighbor)

            ratio = float(getattr(args, "neighbor_sample_ratio", 1.0))
            ratio = max(0.0, min(1.0, ratio))
            if ratio < 1.0:
                k = max(1, int(len(idx_neighbor) * ratio))
                idx_neighbor = idx_neighbor[:k]

            max_n = int(getattr(args, "neighbor_max_samples", 0))
            if max_n > 0:
                idx_neighbor = idx_neighbor[:max_n]

            neighbor_accum_steps = max(1, int(getattr(args, "accum_steps_neighbor", 8)))

            optimizer.zero_grad(set_to_none=True)
            accum_count = 0

            for train_sample_num in tqdm(idx_neighbor, desc=f"Neighbor Task (epoch {epoch})"):
                if train_sample_num == 0:
                    continue

                output_neighbor = entityloc_neighbor_list[train_sample_num:train_sample_num + 1]
                if train_sample_num - args.train_history_len < 0:
                    input_list_neighbor = entityloc_neighbor_list[0: train_sample_num]
                else:
                    input_list_neighbor = entityloc_neighbor_list[train_sample_num - args.train_history_len: train_sample_num]

                history_glist_neighbor = [
                    get_subgraph_cached(snap[:, :3], num_nodes, num_rels, use_cuda, args.gpu,
                                        max_cache_size=args.subgraph_cache_size)
                    for snap in input_list_neighbor
                ]

                out_tri = torch.from_numpy(output_neighbor[0][:, :3]).long()
                if use_cuda:
                    out_tri = out_tri.to(args.gpu)

                with autocast(enabled=scaler.is_enabled()):
                    loss_e_n, loss_r_n, loss_static_n, _ = model.get_loss(
                        history_glist_neighbor,
                        out_tri,
                        static_graph,
                        use_cuda,
                        twin_time_h_list=None, twin_space_h_list=None
                    )
                    loss_neighbor = (args.task_weight * loss_e_n +
                                     (1 - args.task_weight) * loss_r_n +
                                     loss_static_n) / neighbor_accum_steps

                if scaler.is_enabled():
                    scaler.scale(loss_neighbor).backward()
                else:
                    loss_neighbor.backward()

                accum_count += 1

                if accum_count % neighbor_accum_steps == 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            if accum_count % neighbor_accum_steps != 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # Main task
        for train_sample_num in tqdm(idx, desc=f"Main Task (epoch {epoch})"):
            if train_sample_num == 0:
                continue

            output_snap = train_list[train_sample_num]
            out_tri = torch.from_numpy(output_snap).long()
            if use_cuda:
                out_tri = out_tri.to(args.gpu)

            if train_sample_num - args.train_history_len < 0:
                input_list = train_list[0: train_sample_num]
                input_times = train_times[0: train_sample_num]
            else:
                input_list = train_list[train_sample_num - args.train_history_len: train_sample_num]
                input_times = train_times[train_sample_num - args.train_history_len: train_sample_num]

            merged_input_list = [
                maybe_augment_snapshot(
                    s, t, entityloc_by_time,
                    getattr(args, 'merge_locatedin_into_main', False),
                    getattr(args, 'bridge_entity_to_grid', False),
                    rel_locatedin=256
                )
                for s, t in zip(input_list, input_times)
            ]
            history_glist = [
                get_subgraph_cached(snap, num_nodes, num_rels, use_cuda, args.gpu,
                                    max_cache_size=args.subgraph_cache_size)
                for snap in merged_input_list
            ]

            if args.ablation == "no_twin":
                twin_time_h_list = None
            else:
                twin_time_h_list = get_twin_embeddings(
                    input_list, input_times, history_memory,
                    args.twin_period, num_nodes, args.n_hidden, use_cuda, args.gpu
                )

            twin_space_h_list = None
            if getattr(args, 'use_spatial_twin', False) and entityloc_by_time is not None:
                twin_space_h_list = get_spatial_twin_embeddings(
                    input_times, entityloc_by_time, spatial_memory_grid,
                    num_nodes, args.n_hidden, use_cuda, args.gpu,
                    grid_adj=grid_adj, use_grid_prop=getattr(args, 'use_grid_prop', False)
                )

            # build all_triples (add inverse)
            triples = out_tri
            inverse_triples = triples[:, [2, 1, 0]]
            inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
            all_triples = torch.cat([triples, inverse_triples], dim=0)

            evolve_embs, _, r_emb, _, _ = model(
                history_glist, static_graph, use_cuda,
                twin_time_h_list=twin_time_h_list,
                twin_space_h_list=twin_space_h_list
            )
            pre_emb = F.normalize(evolve_embs[-1]) if args.layer_norm else evolve_embs[-1]

            if len(input_times) > 0:
                _mem_put(input_times[-1], pre_emb)

                loss_ent = torch.zeros((), device=pre_emb.device)
                loss_rel = torch.zeros((), device=pre_emb.device)
                loss_twin_val = torch.zeros((), device=pre_emb.device)

                if args.entity_prediction:
                    scores_ob = model.decoder_ob.forward(pre_emb, r_emb, all_triples, mode="train").view(-1, num_nodes)
                    loss_ent = model.loss_e(scores_ob, all_triples[:, 2])

                if args.relation_prediction and args.ablation != "no_rel_pred":
                    score_rel = model.rdecoder.forward(pre_emb, r_emb, all_triples, mode="train").view(-1, 2 * num_rels)
                    loss_rel = model.loss_r(score_rel, all_triples[:, 1])

                if (twin_time_h_list is not None) and (args.ablation != "no_twin") and len(twin_time_h_list) > 0:
                    twin_target = twin_time_h_list[-1].to(pre_emb.device)
                    loss_twin_val = args.twin_loss_weight * F.mse_loss(pre_emb, twin_target)

                if args.ablation == "no_rel_pred":
                    loss = loss_ent + loss_twin_val
                else:
                    loss = args.task_weight * loss_ent + (1 - args.task_weight) * loss_rel + loss_twin_val
            else:
                # rare case: no history
                loss = torch.zeros((), device=pre_emb.device)
                loss_ent = loss
                loss_rel = loss
                loss_twin_val = loss

            losses.append(float(loss.detach().cpu()))
            losses_e.append(float(loss_ent.detach().cpu()))
            losses_r.append(float(loss_rel.detach().cpu()))
            losses_twin.append(float(loss_twin_val.detach().cpu()))

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        lr_now = optimizer.param_groups[0]['lr']
        print_epoch_info(epoch, args.n_epochs, np.mean(losses), np.mean(losses_e),
                         np.mean(losses_r), np.mean(losses_twin), lr_now, best_metric_val)

        # Validation
        if (epoch % args.evaluate_every) == 0:
            metrics = test(
                model,
                history_list=train_list, history_times=train_times,
                test_list=valid_list, test_times=valid_times,
                num_rels=num_rels, num_nodes=num_nodes, use_cuda=use_cuda,
                all_ans_list=all_ans_list_valid, all_ans_r_list=all_ans_list_r_valid,
                model_name=model_state_file, static_graph=static_graph,
                mode="valid", args=args,
                entityloc_by_time=entityloc_by_time,
                grid_adj=grid_adj
            )

            print_valid_table(metrics)

            selector = build_metric_selector(metrics)
            if args.best_metric not in selector:
                raise ValueError(f"Unknown best-metric={args.best_metric}. Available: {sorted(selector.keys())}")
            current_metric = float(selector[args.best_metric])

            scheduler.step(current_metric)
            early_stopping(current_metric, epoch)

            # save if improved
            if current_metric > best_metric_val:
                best_metric_val = current_metric
                best_epoch = epoch
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                print(f"\n✓ Saved BEST checkpoint @epoch={epoch} | {args.best_metric}={best_metric_val:.6f}\n")

            logger.log_epoch(epoch, np.mean(losses), np.mean(losses_e), np.mean(losses_r),
                             np.mean(losses_twin), lr_now, best_metric_val, args.best_metric)
            logger.log_validation(metrics)

            if early_stopping.early_stop:
                print(f"\n🛑 Training stopped early at epoch {epoch}")
                break

    print(f"\n🏁 Best checkpoint: epoch={best_epoch} | {args.best_metric}={best_metric_val:.6f}")
    print(f"📌 ckpt path: {model_state_file}\n")

    # Final test (load best checkpoint)
    test_metrics = test(
        model,
        history_list=(train_list + valid_list), history_times=(train_times + valid_times),
        test_list=test_list, test_times=test_times,
        num_rels=num_rels, num_nodes=num_nodes, use_cuda=use_cuda,
        all_ans_list=all_ans_list_test, all_ans_r_list=all_ans_list_r_test,
        model_name=model_state_file, static_graph=static_graph,
        mode="test", args=args,
        entityloc_by_time=entityloc_by_time,
        grid_adj=grid_adj
    )

    logger.save_all()
    print(f"📊 Training logs saved to ../logs/\n")

    return test_metrics

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TG-GEN: GN-GCN with Twin-Guided Mechanism')

    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "no_twin", "no_timegate", "no_static", "no_neighbor", "no_rel_pred"])
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("--test", action='store_true', default=False)
    parser.add_argument("--run-analysis", action='store_true', default=False)

    parser.add_argument("--multi-step", action='store_true', default=False)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--add-static-graph", action='store_true', default=False)
    parser.add_argument("--relation-evaluation", action='store_true', default=False)

    parser.add_argument("--twin-period", type=int, default=12)
    parser.add_argument("--twin-warmup-margin", type=int, default=5)
    parser.add_argument("--twin-loss-weight", type=float, default=0.05)
    parser.add_argument("--subgraph-cache-size", type=int, default=5000)

    # Spatial Twin / Merge / Bridge
    parser.add_argument("--use-spatial-twin", action="store_true", default=False)
    parser.add_argument("--use-grid-prop", action="store_true", default=False)
    parser.add_argument("--spatial-memory-size", type=int, default=50000)
    parser.add_argument("--spatial-update-neighbors", action="store_true", default=False)

    parser.add_argument("--merge-locatedin-into-main", action="store_true", default=False)
    parser.add_argument("--bridge-entity-to-grid", action="store_true", default=False)

    # Loss weights
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--task-weight", type=float, default=0.7)
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--angle", type=int, default=10)

    # Model hyperparams
    parser.add_argument("--encoder", type=str, default="uvrgcn")
    parser.add_argument("--aggregation", type=str, default="none")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--skip-connect", action='store_true', default=False)
    parser.add_argument("--n-hidden", type=int, default=200)
    parser.add_argument("--opn", type=str, default="sub")
    parser.add_argument("--n-bases", type=int, default=100)
    parser.add_argument("--n-basis", type=int, default=100)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--self-loop", action='store_true', default=False)
    parser.add_argument("--layer-norm", action='store_true', default=False)
    parser.add_argument("--relation-prediction", action='store_true', default=False)
    parser.add_argument("--entity-prediction", action='store_true', default=False)

    # Train
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--evaluate-every", type=int, default=1)

    # Decoder
    parser.add_argument("--decoder", type=str, default="convtranse")
    parser.add_argument("--input-dropout", type=float, default=0.2)
    parser.add_argument("--hidden-dropout", type=float, default=0.2)
    parser.add_argument("--feat-dropout", type=float, default=0.2)

    # History
    parser.add_argument("--train-history-len", type=int, default=3)
    parser.add_argument("--test-history-len", type=int, default=3)
    parser.add_argument("--dilate-len", type=int, default=1)

    # Grid-search (compat)
    parser.add_argument("--grid-search", action='store_true', default=False)
    parser.add_argument("-tune", "--tune", type=str, default="n_hidden,n_layers,dropout,n_bases")

    # Neighbor task knobs
    parser.add_argument("--accum-steps-neighbor", type=int, default=8)
    parser.add_argument("--neighbor-epochs", type=int, default=0)
    parser.add_argument("--neighbor-sample-ratio", type=float, default=1.0)
    parser.add_argument("--neighbor-max-samples", type=int, default=0)

    # BEST METRIC SELECTOR (NEW)
    parser.add_argument("--best-metric", type=str, default="hits1_filter",
                        choices=[
                            # entity
                            "mrr_raw", "mrr_filter",
                            "hits1_raw", "hits3_raw", "hits10_raw",
                            "hits1_filter", "hits3_filter", "hits10_filter",
                            # relation
                            "mrr_raw_r", "mrr_filter_r",
                            "hits1_raw_r", "hits3_raw_r", "hits10_raw_r",
                            "hits1_filter_r", "hits3_filter_r", "hits10_filter_r",
                        ],
                        help="Which validation metric to select best checkpoint (maximization).")

    args = parser.parse_args()
    print(args)

    if args.grid_search:
        out_log = '{}.{}.gs'.format(args.dataset, args.encoder + "-" + args.decoder)
        with open(out_log, 'w') as o_f:
            o_f.write("** Grid Search **\n")

        hyperparameters = args.tune.split(',')
        if args.tune == '' or len(hyperparameters) < 1:
            print("No hyperparameter specified.")
            sys.exit(0)

        grid = hp_range[hyperparameters[0]]
        for hp in hyperparameters[1:]:
            grid = itertools.product(grid, hp_range[hp])
        grid = list(grid)

        print('* {} hyperparameter combinations to try'.format(len(grid)))
        with open(out_log, 'a') as o_f:
            o_f.write('* {} hyperparameter combinations to try\n'.format(len(grid)))

        for i, grid_entry in enumerate(list(grid)):
            if not isinstance(grid_entry, (list, tuple)):
                grid_entry = [grid_entry]
            grid_entry = utils.flatten(grid_entry)

            print('* Hyperparameter Set {}:'.format(i))
            print(grid_entry)

            with open(out_log, 'a') as o_f:
                o_f.write('* Hyperparameter Set {}:\n'.format(i))
                o_f.write("\t".join([str(_) for _ in grid_entry]) + "\n")

            run_experiment(args, grid_entry[0], grid_entry[1], grid_entry[2], grid_entry[3])
    else:
        run_experiment(args)