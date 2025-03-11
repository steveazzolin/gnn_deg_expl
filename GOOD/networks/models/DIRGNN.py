r"""
The implementation of `Discovering Invariant Rationales for Graph Neural Networks <https://openreview.net/pdf?id=hGXij5rfiHw>`_.
"""

import copy
import math

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import BatchNorm
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .GINvirtualnode import vFeatExtractor
from .GINs import FeatExtractor
from torch_geometric.utils.loop import add_self_loops, remove_self_loops



@register.model_register
class DIR(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(DIR, self).__init__(config)

        self.att_net = CausalAttNet(config.ood.ood_param, config)

        config_fe = copy.deepcopy(config)
        config_fe.model.model_layer = config.model.model_layer - 2
        print(f"feat_encoder with {config_fe.model.model_layer} layers")

        self.feat_encoder = FeatExtractor(config_fe, without_embed=True)

        self.num_tasks = config.dataset.num_classes
        self.causal_lin = torch.nn.Linear(config.model.dim_hidden, self.num_tasks)
        self.conf_lin = torch.nn.Linear(config.model.dim_hidden, self.num_tasks)
        self.edge_mask = None

    def forward(self, *args, **kwargs):
        r"""
        The DIR model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            Label predictions and other results for loss calculations.

        """
        data = kwargs.get('data')
        batch_size = data.batch[-1].item() + 1
        # data.edge_index, data.edge_attr = add_self_loops(*remove_self_loops(data.edge_index, data.edge_attr), num_nodes=data.x.shape[0])

        (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch), \
        (conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch), \
        scores = self.att_net(*args, **kwargs)



        # --- Causal repr ---
        set_masks(causal_edge_weight, self)
        causal_rep = self.get_graph_rep(
            data=Data(x=causal_x, edge_index=causal_edge_index,
                      edge_attr=causal_edge_attr, batch=causal_batch),
            batch_size=batch_size
        )
        causal_out = self.get_causal_pred(causal_rep)
        clear_masks(self)
        self.edge_mask = causal_edge_index


        if self.training:
            # --- Conf repr ---
            set_masks(conf_edge_weight, self)
            conf_rep = self.get_graph_rep(
                data=Data(x=conf_x, edge_index=conf_edge_index,
                          edge_attr=conf_edge_attr, batch=conf_batch),
                batch_size=batch_size
            ).detach()
            conf_out = self.get_conf_pred(conf_rep)
            clear_masks(self)

            # --- combine to causal phase (detach the conf phase) ---
            rep_out = None
            # rep_out = []
            # for idx, conf in enumerate(conf_rep):
            #     rep_out.append(self.get_comb_pred(causal_rep, conf))
            # rep_out = torch.stack(rep_out, dim=0)

            # --- Efficient version ---
            rep_out2 = torch.transpose(
                self.get_comb_pred_eff(causal_rep, conf_rep),
                0,
                1
            ) # rep_out2[i] contains a tensor with every causal_rep keeping fixed conf_rep[i]
            
            # DEBUG EFFICIENT VERSION
            # assert torch.allclose(rep_out, rep_out2, atol=1e-6)

            return (rep_out, rep_out2), causal_out, conf_out, scores
        else:
            return causal_out

    def get_graph_rep(self, *args, **kwargs):
        return self.feat_encoder(*args, **kwargs)

    def get_causal_pred(self, h_graph):
        return self.causal_lin(h_graph)

    def get_conf_pred(self, conf_graph_x):
        return self.conf_lin(conf_graph_x)

    def get_comb_pred(self, causal_graph_x, conf_graph_x):
        causal_pred = self.causal_lin(causal_graph_x)
        conf_pred = self.conf_lin(conf_graph_x).detach()
        return torch.sigmoid(conf_pred) * causal_pred
    
    def get_comb_pred_eff(self, causal_graph_x, conf_graph_x):
        causal_pred = self.causal_lin(causal_graph_x)
        conf_pred = self.conf_lin(conf_graph_x).detach()
        return torch.sigmoid(conf_pred).unsqueeze(0) * causal_pred.unsqueeze(1)

@register.model_register
class DIRvGIN(DIR):
    r"""
    The GIN virtual node version of DIR.
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(DIRvGIN, self).__init__(config)
        assert False
        self.att_net = CausalAttNet(config.ood.ood_param, config, virtual_node=True)
        config_fe = copy.deepcopy(config)
        config_fe.model.model_layer = config.model.model_layer - 2
        self.feat_encoder = vFeatExtractor(config_fe, without_embed=True)

@register.model_register
class DIRvGINNB(DIR):
    r"""
    The GIN virtual node without batchnorm version of DIR.
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(DIRvGINNB, self).__init__(config)
        assert False
        self.att_net = CausalAttNet(config.ood.ood_param, config, virtual_node=True, no_bn=True)
        config_fe = copy.deepcopy(config)
        config_fe.model.model_layer = config.model.model_layer - 2
        self.feat_encoder = vFeatExtractor(config_fe, without_embed=True)


class CausalAttNet(nn.Module):
    r"""
    Causal Attention Network adapted from https://github.com/wuyxin/dir-gnn.
    """

    def __init__(self, causal_ratio, config, **kwargs):
        super(CausalAttNet, self).__init__()

        config_catt = copy.deepcopy(config)
        config_catt.model.model_layer = 2
        config_catt.model.dropout_rate = 0

        if kwargs.get('virtual_node'):
            assert False, "Virtual node not in use"
            self.gnn_node = vFeatExtractor(config_catt, without_readout=True, **kwargs)
        else:
            self.gnn_node = FeatExtractor(config_catt, without_readout=True, **kwargs)
        
        # self.linear = nn.Linear(config_catt.model.dim_hidden * 2, 1)
        self.extractor = ExtractorMLP(config)
        self.ratio = causal_ratio

        print("Causal ratio = ", self.ratio)

    def forward(self, *args, **kwargs):
        data = kwargs.get('data') or None

        # extract node embeddigns
        x = self.gnn_node(*args, **kwargs)

        # row, col = data.edge_index
        # edge_rep = torch.cat([x[row], x[col]], dim=-1)
        # edge_score = self.linear(edge_rep).view(-1)

        # WARNING this are log_logits
        edge_score = self.extractor(x, data.edge_index, data.batch).view(-1)

        if data.edge_index.shape[1] != 0:
            (causal_edge_index, causal_edge_attr, causal_edge_weight), \
                (conf_edge_index, conf_edge_attr, conf_edge_weight) = split_graph(data, edge_score, self.ratio)


            # WARNING: Using confounded embeddings?
            causal_x, causal_edge_index, causal_batch, _ = relabel(x, causal_edge_index, data.batch)
            conf_x, conf_edge_index, conf_batch, _ = relabel(x, conf_edge_index, data.batch)
        else:
            raise ValueError(f"{data.x.shape} {data.edge_index.shape}")
            causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch = \
                x, data.edge_index, data.edge_attr, \
                float('inf') * torch.ones(data.edge_index.shape[1], device=data.x.device), \
                data.batch
            conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch = None, None, None, None, None

        return (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch), \
               (conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch), \
               edge_score

class ExtractorMLP(nn.Module):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__()
        hidden_size = config.model.dim_hidden
        self.learn_edge_att = config.ood.extra_param[0]
        dropout_p = config.model.dropout_rate

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits


class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch):
        for module in self._modules.values():
            inputs = module(inputs)
        return inputs


class MLP(BatchSequential):
    def __init__(self, channels, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                m.append(BatchNorm(channels[i]))
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)


def set_masks(mask: Tensor, model: nn.Module):
    r"""
    Adopted from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = True
            module._explain = True
            module.__edge_mask__ = mask
            module._edge_mask = mask


def clear_masks(model: nn.Module):
    r"""
    Adopted from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module._explain = False
            module.__edge_mask__ = None
            module._edge_mask = None


def split_graph_eff(data, edge_score, ratio):
    has_edge_attr = hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None

    causal_edge_index = torch.LongTensor([[],[]]).to(data.x.device)
    causal_edge_weight = torch.tensor([]).to(data.x.device)
    causal_edge_attr = torch.tensor([]).to(data.x.device)
    conf_edge_index = torch.LongTensor([[],[]]).to(data.x.device)
    conf_edge_weight = torch.tensor([]).to(data.x.device)
    conf_edge_attr = torch.tensor([]).to(data.x.device)

    edge_indices, _, _, num_edges, cum_edges = split_batch(data)
    for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):
        n_reserve = int(ratio * N)        
        single_mask = edge_score[C:C+N]

        # Use torch.topk to get the indices of the top n_reserve values
        idx_reserve = torch.topk(single_mask, n_reserve, largest=True).indices
        idx_drop = torch.topk(single_mask, N - n_reserve, largest=False).indices

        causal_edge_index = torch.cat([causal_edge_index, edge_index[:, idx_reserve]], dim=1)
        conf_edge_index = torch.cat([conf_edge_index, edge_index[:, idx_drop]], dim=1)

        causal_edge_weight = torch.cat([causal_edge_weight, single_mask[idx_reserve]])
        conf_edge_weight = torch.cat([conf_edge_weight, -1 * single_mask[idx_drop]])

        if has_edge_attr:
            edge_attr = data.edge_attr[C:C+N]
            causal_edge_attr = torch.cat([causal_edge_attr, edge_attr[idx_reserve]])
            conf_edge_attr = torch.cat([conf_edge_attr, edge_attr[idx_drop]])
        else:
            new_causal_edge_attr = None
            new_conf_edge_attr = None

    return (causal_edge_index, causal_edge_attr, causal_edge_weight), \
           (conf_edge_index, conf_edge_attr, conf_edge_weight)

def split_graph(data, edge_score, ratio):
    r"""
    Adapted from https://github.com/wuyxin/dir-gnn.
    """
    has_edge_attr = hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None


    new_idx_reserve, new_idx_drop, _, _, _ = sparse_topk(edge_score, data.batch[data.edge_index[0]], ratio, descending=True)
    new_causal_edge_index = data.edge_index[:, new_idx_reserve]
    new_conf_edge_index = data.edge_index[:, new_idx_drop]

    new_causal_edge_weight = edge_score[new_idx_reserve]
    new_conf_edge_weight = - edge_score[new_idx_drop]

    if has_edge_attr:
        new_causal_edge_attr = data.edge_attr[new_idx_reserve]
        new_conf_edge_attr = data.edge_attr[new_idx_drop]
    else:
        new_causal_edge_attr = None
        new_conf_edge_attr = None

    return (new_causal_edge_index, new_causal_edge_attr, new_causal_edge_weight), \
           (new_conf_edge_index, new_conf_edge_attr, new_conf_edge_weight)


def split_batch(g):
    r"""
    Adopted from https://github.com/wuyxin/dir-gnn.
    """
    split = degree(g.batch[g.edge_index[0]], dtype=torch.long).tolist()
    edge_indices = torch.split(g.edge_index, split, dim=1)
    num_nodes = degree(g.batch, dtype=torch.long)
    cum_nodes = torch.cat([g.batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]])
    num_edges = torch.tensor([e.size(1) for e in edge_indices], dtype=torch.long).to(g.x.device)
    cum_edges = torch.cat([g.batch.new_zeros(1), num_edges.cumsum(dim=0)[:-1]])

    return edge_indices, num_nodes, cum_nodes, num_edges, cum_edges


def relabel(x, edge_index, batch, pos=None):
    r"""
    Adopted from https://github.com/wuyxin/dir-gnn.
    """
    num_nodes = x.size(0)
    sub_nodes = torch.unique(edge_index)
    x = x[sub_nodes]
    batch = batch[sub_nodes]
    row, col = edge_index
    # remapping the nodes in the explanatory subgraph to new ids.
    node_idx = row.new_full((num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
    edge_index = node_idx[edge_index]
    if pos is not None:
        pos = pos[sub_nodes]
    return x, edge_index, batch, pos


def sparse_sort(src: torch.Tensor, index: torch.Tensor, dim=0, descending=False, eps=1e-12):
    r'''
    Adopted from https://github.com/rusty1s/pytorch_scatter/issues/48.
    '''
    f_src = src.float()
    f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
    norm = (f_src - f_min) / (f_max - f_min + eps) + index.float() * (-1) ** int(descending)
    perm = norm.argsort(dim=dim, descending=descending)

    return src[perm], perm


def sparse_topk(src: torch.Tensor, index: torch.Tensor, ratio: float, dim=0, descending=False, eps=1e-12):
    r"""
    Sparse topk calculation.
    """
    rank, perm = sparse_sort(src, index, dim, descending, eps)
    num_nodes = degree(index, dtype=torch.long)
    k = (ratio * num_nodes.to(float)).ceil().to(torch.long)
    start_indices = torch.cat([torch.zeros((1, ), device=src.device, dtype=torch.long), num_nodes.cumsum(0)])
    mask = [torch.arange(k[i], dtype=torch.long, device=src.device) + start_indices[i] for i in range(len(num_nodes))]
    mask = torch.cat(mask, dim=0)
    mask = torch.zeros_like(index, device=index.device).index_fill(0, mask, 1).bool()
    topk_perm = perm[mask]
    exc_perm = perm[~mask]

    return topk_perm, exc_perm, rank, perm, mask