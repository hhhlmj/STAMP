"""
模型测试脚本 - 显示 MRR 和 Hit@1、Hit@3、Hit@10 指标
"""

import argparse
import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# 添加上一级目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from rgcn import utils
from rgcn.utils import build_sub_graph
from src.rrgcn import RecurrentRGCN
from src.main import get_subgraph_cached
from rgcn.knowledge_graph import _read_triplets_as_list


def test_model(args):
    """
    测试训练好的模型并显示评估指标
    """
    print("\n" + "="*100)
    print("🧪 MODEL EVALUATION")
    print("="*100 + "\n")
    
    # 加载数据
    print("📂 Loading data...")
    data = utils.load_data(args.dataset)
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)

    num_nodes = data.num_nodes
    num_rels = data.num_rels

    # 加载所有答案用于过滤评估
    print("📊 Loading evaluation answers...")
    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)

    # 模型名称和文件路径
    model_name = "{}-{}-{}-ly{}-dilate{}-his{}-weight{}-discount{}-angle{}-dp{}{}{}{}-gpu{}-twin{}"\
        .format(args.dataset, args.encoder, args.decoder, args.n_layers, args.dilate_len, args.train_history_len, 
                args.weight, args.discount, args.angle, args.dropout, args.input_dropout, args.hidden_dropout, 
                args.feat_dropout, args.gpu, args.twin_period)
    model_state_file = '../models/{}'.format(model_name)

    # 检查模型文件是否存在
    if not os.path.exists(model_state_file):
        print(f"❌ 错误：模型文件不存在！")
        print(f"   期望路径：{os.path.abspath(model_state_file)}")
        print(f"\n💡 可用的模型文件：")
        models_dir = '../models'
        for f in os.listdir(models_dir):
            print(f"   - {f}")
        sys.exit(1)

    print(f"✅ 找到模型文件：{model_state_file}")

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    # 静态图处理
    if args.add_static_graph:
        static_triples = np.array(_read_triplets_as_list("../data/" + args.dataset + "/e-w-graph.txt", {}, {}, load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_nodes
        static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().cuda(args.gpu) \
            if use_cuda else torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long()
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None

    # 创建模型
    print("\n🏗️  Building model...")
    model = RecurrentRGCN(args.decoder,
                          args.encoder,
                          num_nodes,
                          num_rels,
                          num_static_rels,
                          num_words,
                          args.n_hidden,
                          args.opn,
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
                          ablation=args.ablation)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    if args.add_static_graph:
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)

    # 加载模型权重
    print("📥 Loading model weights...")
    if use_cuda:
        checkpoint = torch.load(model_state_file, map_location=torch.device(args.gpu))
    else:
        checkpoint = torch.load(model_state_file, map_location=torch.device('cpu'))
    
    model.load_state_dict(checkpoint['state_dict'])
    best_epoch = checkpoint.get('epoch', 'unknown')
    print(f"✅ Model loaded from epoch: {best_epoch}")

    model.eval()

    # 准备测试历史数据
    input_list = [snap for snap in train_list[-args.test_history_len:] + valid_list[-args.test_history_len:]]

    print("\n" + "="*100)
    print("🧪 TESTING ON TEST SET")
    print("="*100 + "\n")

    ranks_raw, ranks_filter = [], []
    ranks_raw_r, ranks_filter_r = [], []

    # 运行测试
    with torch.no_grad():
        for time_idx, test_snap in enumerate(tqdm(test_list, desc="Evaluating")):
            history_glist = [get_subgraph_cached(g, num_nodes, num_rels, use_cuda, args.gpu) for g in input_list]
            test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
            if use_cuda:
                test_triples_input = test_triples_input.to(args.gpu)

            test_triples, final_score, final_r_score = model.predict(history_glist, num_rels, static_graph, 
                                                                     test_triples_input, use_cuda, twin_h_list=None)

            # 实体预测评估
            mrr_filter_snap_r, mrr_snap_r, rank_raw_r, rank_filter_r = utils.get_total_rank(
                test_triples, final_r_score, all_ans_list_r_test[time_idx], eval_bz=1000, rel_predict=1)
            mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(
                test_triples, final_score, all_ans_list_test[time_idx], eval_bz=1000, rel_predict=0)

            ranks_raw.append(rank_raw)
            ranks_filter.append(rank_filter)
            ranks_raw_r.append(rank_raw_r)
            ranks_filter_r.append(rank_filter_r)

            if args.multi_step:
                if not args.relation_evaluation:
                    predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk)
                else:
                    predicted_snap = utils.construct_snap_r(test_triples, num_nodes, num_rels, final_r_score, args.topk)
                if len(predicted_snap):
                    input_list.pop(0)
                    input_list.append(predicted_snap)
            else:
                input_list.pop(0)
                input_list.append(test_snap)

    # 计算指标
    print("\n" + "="*100)
    print("📊 FINAL EVALUATION RESULTS")
    print("="*100 + "\n")

    # 实体预测结果
    print("🎯 ENTITY PREDICTION")
    print("-" * 100)
    mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter = utils.stat_ranks(ranks_filter, "filter_ent")
    print()

    # 关系预测结果
    print("🎯 RELATION PREDICTION")
    print("-" * 100)
    mrr_raw_r = utils.stat_ranks(ranks_raw_r, "raw_rel")
    mrr_filter_r = utils.stat_ranks(ranks_filter_r, "filter_rel")
    print()

    # 汇总表
    print("="*100)
    print("📈 SUMMARY TABLE")
    print("="*100)
    print(f"\n{'Metric':<30} {'MRR (Raw)':<20} {'MRR (Filtered)':<20}")
    print("-" * 100)
    print(f"{'Entity Prediction':<30} {mrr_raw.item():<20.6f} {mrr_filter.item():<20.6f}")
    print(f"{'Relation Prediction':<30} {mrr_raw_r.item():<20.6f} {mrr_filter_r.item():<20.6f}")
    print("=" * 100)

    print("\n✅ 测试完成！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TG-GEN Model Testing')
    
    # 数据集参数
    parser.add_argument("-d", "--dataset", type=str, required=True, help="dataset to use")
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id (-1 for cpu)")
    
    # 模型结构参数
    parser.add_argument("--encoder", type=str, default="uvrgcn", help="encoder name")
    parser.add_argument("--decoder", type=str, default="convtranse", help="decoder name")
    parser.add_argument("--n-hidden", type=int, default=200, help="number of hidden units")
    parser.add_argument("--n-layers", type=int, default=2, help="number of layers")
    parser.add_argument("--n-bases", type=int, default=100, help="number of bases")
    parser.add_argument("--n-basis", type=int, default=100, help="number of basis")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout")
    parser.add_argument("--input-dropout", type=float, default=0.2, help="input dropout")
    parser.add_argument("--hidden-dropout", type=float, default=0.2, help="hidden dropout")
    parser.add_argument("--feat-dropout", type=float, default=0.2, help="feature dropout")
    parser.add_argument("--aggregation", type=str, default="none", help="aggregation method")
    parser.add_argument("--opn", type=str, default="sub", help="operation")
    parser.add_argument("--self-loop", action='store_true', default=False, help="self loop")
    parser.add_argument("--skip-connect", action='store_true', default=False, help="skip connection")
    parser.add_argument("--layer-norm", action='store_true', default=False, help="layer normalization")
    
    # 训练参数
    parser.add_argument("--train-history-len", type=int, default=3, help="training history length")
    parser.add_argument("--test-history-len", type=int, default=3, help="test history length")
    parser.add_argument("--dilate-len", type=int, default=1, help="dilate length")
    parser.add_argument("--weight", type=float, default=1, help="weight for static constraints")
    parser.add_argument("--discount", type=float, default=1, help="discount factor")
    parser.add_argument("--angle", type=int, default=10, help="angle parameter")
    parser.add_argument("--task-weight", type=float, default=0.7, help="task weight")
    
    # 特殊功能
    parser.add_argument("--entity-prediction", action='store_true', default=False, help="entity prediction")
    parser.add_argument("--relation-prediction", action='store_true', default=False, help="relation prediction")
    parser.add_argument("--add-static-graph", action='store_true', default=False, help="add static graph")
    parser.add_argument("--multi-step", action='store_true', default=False, help="multi-step prediction")
    parser.add_argument("--topk", type=int, default=10, help="top-k for multi-step")
    parser.add_argument("--relation-evaluation", action='store_true', default=False, help="evaluate on relation")
    parser.add_argument("--run-analysis", action='store_true', default=False, help="run analysis")
    
    # 孪生机制参数
    parser.add_argument("--twin-period", type=int, default=12, help="twin period")
    parser.add_argument("--ablation", type=str, default="full", help="ablation mode")

    args = parser.parse_args()

    test_model(args)
