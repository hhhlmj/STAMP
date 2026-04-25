#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble evaluator for TG-GEN / GN-GCN style models.

Fixes:
- Avoids UnboundLocalError for test_triples_input by creating tensor unconditionally.
- Compatible with old/new RecurrentRGCN.predict() signatures:
  * may or may not accept twin_h_list keyword
  * may return 3 or 4 values
- Prints final metrics once (MRR, Hits@1/3/10) for raw & filtered.
"""

import argparse
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm

# Path fix so we can import project modules from TG/src/
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from rgcn import utils
from rgcn.utils import build_sub_graph
from src.rrgcn import RecurrentRGCN
from rgcn.knowledge_graph import _read_triplets_as_list


# ---------- Subgraph cache (CPU LRU) ----------
_subgraph_cache = OrderedDict()

def get_subgraph_cached(triples, num_nodes, num_rels, use_cuda, gpu, max_cache_size=50000):
    key = hash(triples.tobytes())
    if key in _subgraph_cache:
        g_cpu = _subgraph_cache.pop(key)
        _subgraph_cache[key] = g_cpu
    else:
        # build on CPU to avoid GPU memory explosion
        g_cpu = build_sub_graph(num_nodes, num_rels, triples, use_cuda=False, gpu=gpu)
        _subgraph_cache[key] = g_cpu
        if len(_subgraph_cache) > max_cache_size:
            _subgraph_cache.popitem(last=False)
    return g_cpu.to(gpu) if (use_cuda and gpu >= 0) else g_cpu


def safe_model_predict(model, history_glist, num_rels, static_graph, test_triplets, use_cuda):
    """
    Compatibility wrapper:
    - Some predict() do not accept twin_h_list kwarg.
    - Some return 3 values (triples, score, score_rel) or 4 (.., embedding).
    """
    try:
        out = model.predict(history_glist, num_rels, static_graph, test_triplets, use_cuda, twin_h_list=None)
    except TypeError:
        out = model.predict(history_glist, num_rels, static_graph, test_triplets, use_cuda)

    if isinstance(out, tuple) and len(out) == 4:
        all_triples, score, score_rel, _ = out
    elif isinstance(out, tuple) and len(out) == 3:
        all_triples, score, score_rel = out
    else:
        raise RuntimeError(f"Unexpected predict() output type/len: {type(out)} / {getattr(out, '__len__', lambda: 'n/a')()}")
    return all_triples, score, score_rel


class EnsembleModel(torch.nn.Module):
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        w = torch.tensor(weights, dtype=torch.float32)
        w = w / w.sum()
        self.register_buffer("weights", w)

    @torch.no_grad()
    def predict(self, history_glist, num_rels, static_graph, test_triplets, use_cuda):
        scores = []
        scores_r = []
        for m in self.models:
            _, s, sr = safe_model_predict(m, history_glist, num_rels, static_graph, test_triplets, use_cuda)
            scores.append(s)
            scores_r.append(sr)
        # weighted average
        w = self.weights.view(-1, 1, 1) if scores[0].dim() == 2 else self.weights.view(-1, 1)
        final_score = torch.sum(torch.stack(scores, dim=0) * w, dim=0)
        w_r = self.weights.view(-1, 1, 1) if scores_r[0].dim() == 2 else self.weights.view(-1, 1)
        final_score_r = torch.sum(torch.stack(scores_r, dim=0) * w_r, dim=0)
        return test_triplets, final_score, final_score_r


def calc_hits_from_ranks(ranks_list, ks=(1, 3, 10)):
    flat = []
    for r in ranks_list:
        if r is None:
            continue
        if torch.is_tensor(r):
            flat.extend(r.view(-1).detach().cpu().numpy().tolist())
        else:
            arr = np.asarray(r).reshape(-1)
            flat.extend(arr.tolist())
    if len(flat) == 0:
        return {k: 0.0 for k in ks}
    ranks = np.asarray(flat, dtype=np.float64)
    return {k: float(np.mean(ranks <= k)) for k in ks}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--dataset", type=str, default="ICEWS18")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--test-history-len", type=int, default=3)
    p.add_argument("--train-history-len", type=int, default=3)
    p.add_argument("--decoder", type=str, default="convtranse")
    p.add_argument("--encoder", type=str, default="uvrgcn")
    p.add_argument("--opn", type=str, default="sub")
    p.add_argument("--n-bases", type=int, default=100)
    p.add_argument("--n-basis", type=int, default=100)
    p.add_argument("--self-loop", action="store_true", default=True)
    p.add_argument("--skip-connect", action="store_true", default=False)
    p.add_argument("--layer-norm", action="store_true", default=True)
    p.add_argument("--input-dropout", type=float, default=0.2)
    p.add_argument("--hidden-dropout", type=float, default=0.2)
    p.add_argument("--feat-dropout", type=float, default=0.2)
    p.add_argument("--aggregation", type=str, default="none")
    p.add_argument("--add-static-graph", action="store_true", default=True)
    p.add_argument("--entity-prediction", action="store_true", default=True)
    p.add_argument("--relation-prediction", action="store_true", default=True)

    p.add_argument("--model1", type=str, required=True, help="path to checkpoint 1 (.pth)")
    p.add_argument("--model2", type=str, required=True, help="path to checkpoint 2 (.pth)")
    p.add_argument("--w1", type=float, default=0.5)
    p.add_argument("--w2", type=float, default=0.5)
    p.add_argument("--cache-size", type=int, default=50000)

    # IMPORTANT: if your checkpoints use different hidden/layers/dropout, pass them here
    p.add_argument("--n-hidden", type=int, default=200)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)

    args = p.parse_args()

    use_cuda = (args.gpu >= 0) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    print("Loading data...")
    data = utils.load_data(args.dataset)
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)
    history_list = train_list + valid_list

    num_nodes = data.num_nodes
    num_rels = data.num_rels

    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)

    # static graph
    if args.add_static_graph:
        static_triples = np.array(_read_triplets_as_list(f"../data/{args.dataset}/e-w-graph.txt", {}, {}, load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_nodes
        static_node_id = torch.from_numpy(np.arange(num_words + num_nodes)).view(-1, 1).long()
        if use_cuda:
            static_node_id = static_node_id.to(args.gpu)
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)
    else:
        num_static_rels, num_words, static_graph = 0, 0, None

    def load_model(ckpt_path: str):
        print(f"Loading model from {ckpt_path} ...")
        m = RecurrentRGCN(
            decoder_name=args.decoder,
            encoder_name=args.encoder,
            num_ents=num_nodes,
            num_rels=num_rels,
            num_static_rels=num_static_rels,
            num_words=num_words,
            h_dim=args.n_hidden,
            opn=args.opn,
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
            use_static=args.add_static_graph,
            entity_prediction=args.entity_prediction,
            relation_prediction=args.relation_prediction,
            use_cuda=use_cuda,
            gpu=args.gpu,
        )
        if use_cuda:
            m.cuda()
            ckpt = torch.load(ckpt_path, map_location=torch.device(args.gpu))
        else:
            ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        m.load_state_dict(state, strict=True)
        m.eval()
        return m

    model1 = load_model(args.model1)
    model2 = load_model(args.model2)
    ensemble = EnsembleModel([model1, model2], weights=[args.w1, args.w2])
    if use_cuda:
        ensemble.cuda()
    ensemble.eval()

    ranks_raw, ranks_filter = [], []
    ranks_raw_r, ranks_filter_r = [], []

    # rolling window init
    input_list = [snap for snap in history_list[-args.test_history_len:]]

    print("\n==============================")
    print("Start Ensemble Testing...")
    print("==============================")
    for time_idx, test_snap in enumerate(tqdm(test_list)):
        history_glist = [get_subgraph_cached(g, num_nodes, num_rels, use_cuda, args.gpu, max_cache_size=args.cache_size)
                         for g in input_list]

        test_triples_input = torch.LongTensor(test_snap)
        if use_cuda:
            test_triples_input = test_triples_input.to(args.gpu)

        test_triples, final_score, final_r_score = ensemble.predict(
            history_glist, num_rels, static_graph, test_triples_input, use_cuda
        )

        # entity metrics
        _, _, rank_raw, rank_filter = utils.get_total_rank(
            test_triples, final_score, all_ans_list_test[time_idx], eval_bz=1000, rel_predict=0
        )
        # relation metrics
        _, _, rank_raw_r, rank_filter_r = utils.get_total_rank(
            test_triples, final_r_score, all_ans_list_r_test[time_idx], eval_bz=1000, rel_predict=1
        )

        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        ranks_raw_r.append(rank_raw_r)
        ranks_filter_r.append(rank_filter_r)

        # update history (one-step)
        input_list.pop(0)
        input_list.append(test_snap)

    # summarize
    mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter = utils.stat_ranks(ranks_filter, "filter_ent")
    mrr_raw_r = utils.stat_ranks(ranks_raw_r, "raw_rel")
    mrr_filter_r = utils.stat_ranks(ranks_filter_r, "filter_rel")

    hits_raw = calc_hits_from_ranks(ranks_raw, ks=(1, 3, 10))
    hits_filter = calc_hits_from_ranks(ranks_filter, ks=(1, 3, 10))
    hits_raw_r = calc_hits_from_ranks(ranks_raw_r, ks=(1, 3, 10))
    hits_filter_r = calc_hits_from_ranks(ranks_filter_r, ks=(1, 3, 10))

    print("\n" + "=" * 70)
    print(f"FINAL ENSEMBLE RESULT ({args.dataset})  weights=({args.w1:.2f},{args.w2:.2f})")
    print("=" * 70)
    print("Entity Prediction:")
    print(f"  MRR raw   {mrr_raw:.4f} | H@1 {hits_raw[1]:.4f} H@3 {hits_raw[3]:.4f} H@10 {hits_raw[10]:.4f}")
    print(f"  MRR filt  {mrr_filter:.4f} | H@1 {hits_filter[1]:.4f} H@3 {hits_filter[3]:.4f} H@10 {hits_filter[10]:.4f}")
    print("Relation Prediction:")
    print(f"  MRR raw   {mrr_raw_r:.4f} | H@1 {hits_raw_r[1]:.4f} H@3 {hits_raw_r[3]:.4f} H@10 {hits_raw_r[10]:.4f}")
    print(f"  MRR filt  {mrr_filter_r:.4f} | H@1 {hits_filter_r[1]:.4f} H@3 {hits_filter_r[3]:.4f} H@10 {hits_filter_r[10]:.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
