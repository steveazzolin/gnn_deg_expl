import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import is_undirected, to_undirected, degree, coalesce
from torch_geometric import __version__ as pyg_v

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.train import lift_node_att_to_edge_att
from GOOD.utils.splitting import split_graph, relabel
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GINs import FeatExtractor
import copy

from torch_geometric.nn import InstanceNorm, BatchNorm


@register.model_register
class GiSST(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GiSST, self).__init__(config)

        config = copy.deepcopy(config)
        fe_kwargs = {'mitigation_readout': config.mitigation_readout}

        self.learn_edge_att = config.ood.extra_param[0]
        self.config = config
        self.edge_mask = None
        print("Using mitigation_expl_scores:", config.mitigation_expl_scores)

        if config.mitigation_sampling == "raw":
            fe_kwargs["gnn_clf_layer"] = config.model.gnn_clf_layer
            print(f"Using mitigation_sampling==raw with {config.model.gnn_clf_layer} layers")

        self.gnn_clf = FeatExtractor(config, **fe_kwargs)
        self.classifier = Classifier(config)

        self.prob_mask = ProbMask(config.dataset.dim_node)
        self.extractor = AttentionProb(config.dataset.dim_node, config)


    
    def forward(self, *args, **kwargs):
        """
        Forward pass.

        Args:
            x (torch.float): Node feature tensor with shape [num_nodes, num_node_feat].
            edge_index (torch.long): Edges in COO format with shape [2, num_edges].
            edge_weight (torch.float): Weight for each edge with shape [num_edges].
            return_probs (boolean): Whether to return x_prob and edge_prob.
            batch (None or torch.long): Node assignment for a batch of graphs with shape 
                [num_nodes] for graph classification. None for node classification.

        Return:
            x (torch.float): Final output of the network with shape 
                [num_nodes, output_size].
            x_prob (torch.float): Node feature probability with shape [input_size].
            edge_prob (torch.float): Edge probability with shape [num_edges].
        """
        data = kwargs.get('data')

        # node feature explanation
        x_prob = self.prob_mask()
        x_prob.requires_grad_()
        x_prob.retain_grad()

        if self.config.dataset.dataset_type == 'mol':
            # if node features are needed to compute AtomEmbedding, apply attention later only to extract edge scores
            scaled_x = data.x * x_prob
        else:
            data.x = data.x * x_prob
            scaled_x = data.x

        # topological explanation        
        att_log_prob = self.extractor(scaled_x, data.edge_index, data.batch)
       
        att_prob = torch.sigmoid(att_log_prob)
        att_prob = torch.clamp(att_prob, self.extractor.clamp_min, self.extractor.clamp_max)
        att_prob.requires_grad_()
        att_prob.retain_grad()

        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                if self.config.average_edge_attn == "default":
                    assert False, f"self.config.average_edge_attn == 'default' not in use"
                else:
                    data.ori_edge_index = data.edge_index.detach().clone() #for backup and debug
                    data.edge_index, edge_att = to_undirected(data.edge_index, att_prob.squeeze(-1), reduce="mean")

                    if not data.edge_attr is None:
                        assert False, "edge_attr not in use"
                        edge_index_sorted, edge_attr_sorted = coalesce(data.ori_edge_index, data.edge_attr, is_sorted=False)
                        data.edge_attr = edge_attr_sorted    
            else:
                edge_att = att_prob
        else:
            edge_att = lift_node_att_to_edge_att(att_prob, data.edge_index)

        
        if kwargs.get('weight', None):
            if kwargs.get('is_ratio'):
                (causal_edge_index, causal_edge_attr, causal_edge_weight), _ = split_graph(data, edge_att, kwargs.get('weight'))
                causal_x, causal_edge_index, causal_batch, _ = relabel(data.x, causal_edge_index, data.batch)
                data.x = causal_x
                data.batch = causal_batch
                data.edge_index = causal_edge_index
                if not data.edge_attr is None:
                    data.edge_attr = causal_edge_attr
                edge_att = causal_edge_weight                
            else:
                data.edge_index = (data.edge_index.T[edge_att >= kwargs.get('weight')]).T
                if not data.edge_attr is None:
                    data.edge_attr = data.edge_attr[edge_att >= kwargs.get('weight')]
                edge_att = edge_att[edge_att >= kwargs.get('weight')]

        if self.config.mitigation_expl_scores == "topK" or self.config.mitigation_expl_scores == "topk":
            (causal_edge_index, causal_edge_attr, edge_att), \
                _ = split_graph(data, edge_att, self.config.mitigation_expl_scores_topk)
           
            causal_x, causal_edge_index, causal_batch, _ = relabel(data.x, causal_edge_index, data.batch)

            data_topk = Data(x=causal_x, edge_index=causal_edge_index, edge_attr=causal_edge_attr, batch=causal_batch)
            kwargs['data'] = data_topk
            kwargs["batch_size"] =  data.batch[-1].item() + 1
        
        set_masks(edge_att, self, att_prob)
        logits = self.classifier(self.gnn_clf(*args, **kwargs))
        clear_masks(self)
        self.edge_mask = edge_att

        return logits, x_prob, att_prob
    
    @torch.no_grad()
    def probs(self, *args, **kwargs):
        # nodes x classes
        out = self(*args, **kwargs)
        
        if len(out) == 5:
            logits, att, edge_att, _, _ = out
        else:
            logits, att, edge_att = out

        if logits.shape[-1] > 1:
            return logits.softmax(dim=1)
        else:
            return logits.sigmoid()
        
    @torch.no_grad()
    def log_probs(self, eval_kl=False, *args, **kwargs):
        # nodes x classes
        out = self(*args, **kwargs)
        
        if len(out) == 5:
            logits, att, edge_att, _, _ = out
        else:
            logits, att, edge_att = out
            
        if logits.shape[-1] > 1:
            return logits.log_softmax(dim=1)
        else:
            if eval_kl: # make the single logit a proper distribution summing to 1 to compute KL
                logits = logits.sigmoid()
                new_logits = torch.zeros((logits.shape[0], logits.shape[1]+1), device=logits.device)
                new_logits[:, 1] = new_logits[:, 1] + logits.squeeze(1)
                new_logits[:, 0] = 1 - new_logits[:, 1]
                new_logits[new_logits == 0.] = 1e-10
                return new_logits.log()
            else:
                return logits.sigmoid().log()
            
    @torch.no_grad()
    def predict_from_subgraph(self, edge_att=False, log=None, eval_kl=None, node_att=False, *args, **kwargs):
        # node feature explanation
        x_prob = self.prob_mask()
        x_prob.requires_grad_()
        x_prob.retain_grad()
        kwargs['data'].x = kwargs['data'].x * x_prob

        if self.config.dataset.dataset_type == 'mol':
            # if node features are needed to compute AtomEmbedding, apply attention later only to extract edge scores
            scaled_x = kwargs['data'].x * x_prob
        else:
            kwargs['data'].x = kwargs['data'].x * x_prob
            scaled_x = kwargs['data'].x

        set_masks(edge_att, self, node_att)
        lc_logits = self.classifier(self.gnn_clf(*args, **kwargs))
        clear_masks(self)


        if log is None:
            if lc_logits.shape[-1] > 1:
                return lc_logits.argmax(-1)
            else:
                return lc_logits.sigmoid()
        else:
            assert not (eval_kl is None)
            if lc_logits.shape[-1] > 1:
                return lc_logits.log_softmax(dim=1)
            else:
                if eval_kl: # make the single logit a proper distribution summing to 1 to compute KL
                    lc_logits = lc_logits.sigmoid()
                    new_logits = torch.zeros((lc_logits.shape[0], lc_logits.shape[1]+1), device=lc_logits.device)
                    new_logits[:, 1] = new_logits[:, 1] + lc_logits.squeeze(1)
                    new_logits[:, 0] = 1 - new_logits[:, 1]
                    new_logits[new_logits == 0.] = 1e-10
                    return new_logits.log()
                else:
                    return lc_logits.sigmoid().log()

    @torch.no_grad()         
    def get_subgraph(self, ratio=None, *args, **kwargs):
        data = kwargs.get('data') or None
        data.ori_x = data.x

        # node feature explanation
        x_prob = self.prob_mask()
        
        if self.config.dataset.dataset_type == 'mol':
            # if node features are needed to compute AtomEmbedding, apply attention later only to extract edge scores
            scaled_x = data.x * x_prob
        else:
            data.x = data.x * x_prob
            scaled_x = data.x

        # edge topological explanation
        att_log_prob = self.extractor(scaled_x, data.edge_index, data.batch)
        att_prob = torch.sigmoid(att_log_prob)
        att_prob = torch.clamp(att_prob, self.extractor.clamp_min, self.extractor.clamp_max)

        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                if self.config.average_edge_attn == "default":
                    raise NotImplementedError("average_edge_attn='default' not implemented")
                else:
                    data.ori_edge_index = data.edge_index.detach().clone() #for backup and debug

                    if not data.edge_attr is None:
                        edge_index_sorted, edge_attr_sorted = coalesce(data.ori_edge_index, data.edge_attr, is_sorted=False)
                        data.edge_attr = edge_attr_sorted   
                    if hasattr(data, "edge_gt") and not data.edge_gt is None:
                        edge_index_sorted, edge_gt_sorted = coalesce(data.ori_edge_index, data.edge_gt, is_sorted=False)
                        data.edge_gt = edge_gt_sorted
                    if hasattr(data, "causal_mask") and not data.causal_mask is None:
                        _, data.causal_mask = coalesce(data.edge_index, data.causal_mask, is_sorted=False)

                    data.edge_index, edge_att = to_undirected(data.edge_index, att_prob.squeeze(-1), reduce="mean")
            else:
                edge_att = att_prob
        else:
            edge_att = lift_node_att_to_edge_att(att_prob, data.edge_index)
        

        if kwargs.get('return_attn', False):
            assert False, "not in use"
            self.attn_distrib = self.gnn.encoder.get_attn_distrib()
            self.gnn.encoder.reset_attn_distrib()

        edge_att = edge_att.view(-1)
        if ratio is None:
            return edge_att, att_prob
        assert False
        
class ProbMask(torch.nn.Module):
    """
    Probability mask generator.

    Args:
        shape (tuple of int): Shape of the probability mask.
        clamp_min (float): Clamping minimum for probability, for numerical stability.
        clamp_max (float): Clamping maximum for probability, for numerical stability. 
    """
    def __init__(
        self, 
        shape, 
        clamp_min=0.001,
        clamp_max=0.999
    ):
        super(ProbMask, self).__init__()
        self.shape = shape
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.mask_weight = torch.nn.Parameter(torch.randn(shape))

    def forward(self):
        return torch.clamp(
            torch.sigmoid(self.mask_weight), 
            self.clamp_min, 
            self.clamp_max
        )

class AttentionProb(torch.nn.Module):
    """
    Edge attention mechanism for generating sigmoid probability using concatentation of 
    source and target node features.

    Args:
        input_size (int): Number of input node features.
        clamp_min (float): Clamping minimum for the output probability, for numerical
            stability.
        clamp_max (float): Clamping maximum for the output probability, for numerical 
            stability.
    """
    def __init__(
        self, 
        input_size, 
        config,
        clamp_min=0.001,
        clamp_max=0.999
    ):
        super(AttentionProb, self).__init__()
        self.input_size = input_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.learn_edge_att = config.ood.extra_param[0]

        hidden_size = config.model.dim_hidden
        dropout_p = config.model.dropout_rate

        if self.learn_edge_att:
            print("#D#Adopting edge level scores")
        else:
            print("#D#Adopting node level scores")

        if self.learn_edge_att:
            self.feature_extractor = MLP([input_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
        else:
            self.feature_extractor = MLP([input_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p)

        # if self.learn_edge_att:
        #     self.att_weight = torch.nn.Parameter(
        #         torch.randn(input_size * 2)
        #     )
        # else:
        #     self.att_weight = torch.nn.Parameter(
        #         torch.randn(input_size)
        #     )
    
    def forward(
        self,
        x,
        edge_index,
        batch
    ):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = x[col], x[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(x, batch)
        return att_log_logits
    
        # if self.learn_edge_att:
        #     att = torch.matmul(
        #         torch.cat(
        #             (
        #                 x[edge_index[0, :], :], # source node features
        #                 x[edge_index[1, :], :]  # target node features
        #             ), 
        #             dim=1
        #         ), 
        #         self.att_weight
        #     )
        # else:
        #     att = torch.matmul(
        #         x, 
        #         self.att_weight
        #     )
        # return att
    
class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch):
        for module in self._modules.values():
            if isinstance(module, (InstanceNorm)):
                inputs = module(inputs, batch)
            else:
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
    
def set_masks(mask: Tensor, model: nn.Module, node_mask:Tensor=None):
    r"""
    Modified from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            if pyg_v == "2.4.0":
                module._fixed_explain = True
            else:
                module.__explain__ = True
                module._explain = True
            module._apply_sigmoid = False    
            module._edge_mask = mask

            if model.extractor.learn_edge_att == False:
                module._node_mask = node_mask


def clear_masks(model: nn.Module):
    r"""
    Modified from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            if pyg_v == "2.4.0":
                module._fixed_explain = False
            else:
                module.__explain__ = False
                module._explain = False
            module._edge_mask = None
            module._node_mask = None
