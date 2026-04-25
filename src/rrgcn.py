import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from src.model import BaseRGCN
from src.decoder import ConvTransE, ConvTransR


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.05, dim=-1):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            if self.cls > 1:
                true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class TwinGuidedMechanism(nn.Module):
    """
    Learnable gate:
      gate = sigmoid( proj(tanh(W[cur||twin])) )
      fused = (1-gate)*cur + gate*twin

    Compared to dot-gate, this is more robust when spatial/bridge signals are noisy.
    """
    def __init__(self, h_dim: int, gate_dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(h_dim * 2, h_dim)
        self.proj = nn.Linear(h_dim, 1)
        self.drop = nn.Dropout(gate_dropout)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, current_h: torch.Tensor, twin_h: torch.Tensor):
        if twin_h is None:
            return current_h
        x = torch.cat([current_h, twin_h], dim=-1)
        x = torch.tanh(self.fc(x))
        x = self.drop(x)
        gate = torch.sigmoid(self.proj(x))  # [N,1]
        return (1.0 - gate) * current_h + gate * twin_h


class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        sc = (self.skip_connect and idx != 0)
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(
                self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                activation=act, dropout=self.dropout, self_loop=self.self_loop,
                skip_connect=sc, rel_emb=self.rel_emb
            )
        raise NotImplementedError

    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            r = init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')

        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = init_ent_emb[node_id]
        if self.skip_connect:
            prev_h = []
            for layer in self.layers:
                prev_h = layer(g, prev_h)
        else:
            for layer in self.layers:
                layer(g, [])
        return g.ndata.pop('h')


class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words,
                 h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0.0, self_loop=False, skip_connect=False,
                 layer_norm=False, input_dropout=0.0, hidden_dropout=0.0, feat_dropout=0.0,
                 aggregation='cat', weight=1.0, discount=0.0, angle=0.0,
                 use_static=False, entity_prediction=False, relation_prediction=False,
                 use_cuda=False, gpu=0, analysis=False, ablation="full",
                 label_smoothing=0.05):
        super().__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.gpu = gpu
        self.use_cuda = use_cuda
        self.ablation = ablation

        # twin fusion
        self.twin_guided_layer = TwinGuidedMechanism(h_dim, gate_dropout=min(0.2, max(0.0, dropout)))

        # parameters
        self.w1 = nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True)
        nn.init.xavier_normal_(self.w1)

        self.w2 = nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True)
        nn.init.xavier_normal_(self.w2)

        self.emb_rel = nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True)
        nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True)
        nn.init.normal_(self.dynamic_emb)

        # static graph branch
        if self.use_static:
            self.words_emb = nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True)
            nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(
                self.h_dim, self.h_dim, self.num_static_rels * 2, num_bases,
                activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False
            )
            self.static_loss = nn.MSELoss()

        self.loss_r = nn.CrossEntropyLoss()
        self.loss_e = LabelSmoothingLoss(num_ents, smoothing=label_smoothing)

        self.rgcn = RGCNCell(
            num_ents, h_dim, h_dim, num_rels * 2,
            num_bases, num_basis, num_hidden_layers, dropout,
            self_loop, skip_connect, encoder_name, self.opn, self.emb_rel,
            use_cuda, analysis
        )

        # time gate
        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)

        # relation evolve cell
        self.relation_cell_1 = nn.GRUCell(self.h_dim * 2, self.h_dim)

        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(
                num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout,
                channels=200, kernel_size=3
            )
            self.rdecoder = ConvTransR(
                num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout
            )
        else:
            raise NotImplementedError

    def _to_device(self, x):
        if x is None:
            return None
        if self.use_cuda and self.gpu >= 0:
            try:
                return x.to(self.gpu)
            except Exception:
                return x
        try:
            return x.cpu()
        except Exception:
            return x

    def forward(self, g_list, static_graph, use_cuda, twin_time_h_list=None, twin_space_h_list=None):
        gate_list, degree_list = [], []

        if self.use_static and (self.ablation != "no_static"):
            static_graph = self._to_device(static_graph)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        history_embs = []
        for i, g in enumerate(g_list):
            g = self._to_device(g)

            temp_e = self.h[g.r_to_e.long()]
            device = self.h.device
            x_input = torch.zeros(self.num_rels * 2, self.h_dim, device=device)

            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1], :]
                if x.numel() == 0:
                    continue
                x_input[int(r_idx)] = torch.mean(x, dim=0, keepdim=False)

            x_input = torch.cat((self.emb_rel, x_input), dim=1)
            if i == 0:
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel)
            else:
                self.h_0 = self.relation_cell_1(x_input, self.h_0)
            self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0

            current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0])
            current_h = F.normalize(current_h) if self.layer_norm else current_h

            if twin_time_h_list is not None and i < len(twin_time_h_list):
                current_h = self.twin_guided_layer(current_h, twin_time_h_list[i].to(current_h.device))

            if twin_space_h_list is not None and i < len(twin_space_h_list):
                current_h = self.twin_guided_layer(current_h, twin_space_h_list[i].to(current_h.device))

            if self.ablation == "no_timegate":
                self.h = current_h
            else:
                time_weight = torch.sigmoid(torch.mm(self.h, self.time_gate_weight) + self.time_gate_bias)
                self.h = time_weight * current_h + (1.0 - time_weight) * self.h

            history_embs.append(self.h)

        return history_embs, static_emb, self.h_0, gate_list, degree_list

    @torch.no_grad()
    def predict(self, test_graph, num_rels, static_graph, test_triplets, use_cuda,
                twin_time_h_list=None, twin_space_h_list=None):
        if test_triplets is None or len(test_triplets) == 0:
            empty = torch.empty((0, 3), dtype=torch.long, device=self.dynamic_emb.device)
            return empty, empty, empty, self.dynamic_emb

        test_triplets = self._to_device(test_triplets)
        inverse_test_triplets = test_triplets[:, [2, 1, 0]]
        inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels
        all_triples = torch.cat((test_triplets, inverse_test_triplets), dim=0)
        all_triples = self._to_device(all_triples)

        evolve_embs, _, r_emb, _, _ = self.forward(
            test_graph, static_graph, use_cuda,
            twin_time_h_list=twin_time_h_list,
            twin_space_h_list=twin_space_h_list
        )
        embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

        score = self.decoder_ob.forward(embedding, r_emb, all_triples, mode="test")
        score_rel = self.rdecoder.forward(embedding, r_emb, all_triples, mode="test")
        return all_triples, score, score_rel, embedding