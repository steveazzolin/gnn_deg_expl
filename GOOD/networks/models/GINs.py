from typing import Callable, Optional
import copy

import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as gnn
from torch import Tensor
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_scatter.composite import scatter_softmax
import torch.nn.functional as F

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic, BasicEncoder, GINEConv, IdentityConv
from .Classifiers import Classifier
from .MolEncoders import AtomEncoder, BondEncoder
from .Pooling import GlobalAddPool
from torch.nn import Identity



from sklearn.metrics import accuracy_score

@register.model_register
class GIN(GNNBasic):
    r"""
    The Graph Neural Network from the `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.num_classes`, :obj:`config.dataset.dataset_type`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):

        super().__init__(config)
        self.feat_encoder = FeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        r"""
        The GIN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            label predictions

        """
        out_readout = self.feat_encoder(*args, **kwargs)

        out = self.classifier(out_readout)
        return out


class FeatExtractor(GNNBasic):
    r"""
        GIN feature extractor using the :class:`~GINEncoder` or :class:`~GINMolEncoder`.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.dataset_type`)
    """
    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        super(FeatExtractor, self).__init__(config)
        print("#D#Init FeatExtractor")
        num_layer = config.model.model_layer
        if config.dataset.dataset_type == 'mol':
            self.encoder = MolEncoder(config, **kwargs)
            self.edge_feat = True
        else:
            self.encoder = Encoder(config, **kwargs)
            self.edge_feat = False

    def forward(self, *args, **kwargs):
        r"""
        GIN feature extractor using the :class:`~GINEncoder` or :class:`~GINMolEncoder`.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            node feature representations
        """
        if self.edge_feat:
            x, edge_index, edge_attr, batch, batch_size = self.arguments_read(*args, edge_feat=self.edge_feat, **kwargs)
            kwargs.pop('batch_size', 'not found')
            out_readout = self.encoder(x, edge_index, edge_attr, batch, batch_size, **kwargs)
        else:
            x, edge_index, batch, batch_size = self.arguments_read(*args, **kwargs)
            kwargs.pop('batch_size', 'not found')
            out_readout = self.encoder(x, edge_index, batch, batch_size, **kwargs)
        return out_readout

    def get_node_repr(self, *args, **kwargs):

        if self.edge_feat:
            x, edge_index, edge_attr, batch, batch_size = self.arguments_read(*args, edge_feat=self.edge_feat, **kwargs)
            kwargs.pop('batch_size', 'not found')
            node_repr = self.encoder.get_node_repr(x, edge_index, edge_attr, batch, batch_size, **kwargs)
        else:
            x, edge_index, batch, batch_size = self.arguments_read(*args, **kwargs)
            kwargs.pop('batch_size', 'not found')
            node_repr = self.encoder.get_node_repr(x, edge_index, batch, batch_size, **kwargs)
        return node_repr


class Encoder(BasicEncoder):
    r"""
    The GIN encoder for non-molecule data, using the :class:`~GINConv` operator for message passing.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`)
    """

    def __init__(self, config: Union[CommonArgs, Munch], *args, **kwargs):

        super(Encoder, self).__init__(config, *args, **kwargs)
        
        num_layer = config.model.model_layer if kwargs.get('gnn_clf_layer') is None else kwargs.get('gnn_clf_layer')
        print("Num layers =",num_layer)

        self.without_readout = kwargs.get('without_readout')

        self.convs = nn.ModuleList()

        if num_layer == 0:
            self.convs.append(self.get_conv_layer(config, backbone="Identity", without_embed=None, no_bias=None))
        else:
            self.convs = self.convs.extend(
                [
                    self.get_conv_layer(
                        config,
                        backbone=config.model.backbone,
                        without_embed=True if n > 0 else kwargs.get('without_embed'),
                        no_bias=kwargs.get('no_bias', False)
                    )
                        for n in range(num_layer)
                ]
            )

        # self.batch_norms = nn.ModuleList([
        #     gnn.BatchNorm(config.model.dim_hidden if i > 0 else config.dataset.dim_node, track_running_stats=True)
        #         for i in range(num_layer)
        # ])

    def get_attn_distrib(self):
        ret = []
        for conv in self.convs:
            ret.append(conv.attn_distrib)
        return ret

    def reset_attn_distrib(self):
        for conv in self.convs:
            conv.attn_distrib = []

    def forward(self, x, edge_index, batch, batch_size, **kwargs):
        r"""
        The GIN encoder for non-molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            batch (Tensor): batch indicator
            batch_size (int): batch size

        Returns (Tensor):
            graph feature representations
        """

        node_repr = self.get_node_repr(x, edge_index, batch, batch_size, **kwargs)

        if self.without_readout or kwargs.get('without_readout'):
            return node_repr

        out_readout = self.readout(
            node_repr,
            batch,
            batch_size,
            edge_index=edge_index,
            edge_mask=getattr(self.convs[0], "_edge_mask", None),
            node_mask=getattr(self.convs[0], "_node_mask", None)
        )
        return out_readout

    def get_node_repr(self, x, edge_index, batch, batch_size, **kwargs):
        r"""
        The GIN encoder for non-molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            batch (Tensor): batch indicator
            batch_size (int): batch size

        Returns (Tensor):
            node feature representations
        """

        layer_feat = x
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            
            if isinstance(conv, IdentityConv):
                continue

            post_conv = batch_norm(conv(layer_feat, edge_index, batch=batch))

            if i != len(self.convs) - 1:
                post_conv = relu(post_conv)

            layer_feat = dropout(post_conv)
        return layer_feat

class MolEncoder(BasicEncoder):
    r"""The GIN encoder for molecule data, using the :class:`~GINEConv` operator for message passing.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`)
    """

    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        super(MolEncoder, self).__init__(config, **kwargs)

        exit("MolEncoder not in use")

        self.without_readout = kwargs.get('without_readout')
        self.config = config
        num_layer = config.model.model_layer
        print(f"Initing GINMolEncoder for {num_layer} layers")

        if kwargs.get('without_embed'):
            self.atom_encoder = Identity()
        else:
            self.atom_encoder = AtomEncoder(config.model.dim_hidden, config)

        self.convs = nn.ModuleList(
            [
                GINEConv(
                    nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                    self.get_norm_layer(config), 
                                    nn.LeakyReLU(),
                                    nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)
                    ),
                    config,
                    BondEncoder(self.nn[0].in_features if hasattr(self.nn[0], 'in_features') else self.nn[0].in_channels, config)
                )
            ] +
            [
                GINEConv(
                    nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                    self.get_norm_layer(config), 
                                    nn.LeakyReLU(),
                                    nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)
                    ),
                    config,
                    BondEncoder(self.nn[0].in_features if hasattr(self.nn[0], 'in_features') else self.nn[0].in_channels, config)
                )
                for _ in range(num_layer - 1)
            ]
        )

    def forward(self, x, edge_index, edge_attr, batch, batch_size, **kwargs):
        r"""
        The GIN encoder for molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            edge_attr (Tensor): edge attributes
            batch (Tensor): batch indicator
            batch_size (int): Batch size.

        Returns (Tensor):
            graph feature representations
        """
        node_repr = self.get_node_repr(x, edge_index, edge_attr, batch, batch_size, **kwargs)

        if self.without_readout or kwargs.get('without_readout'):
            return node_repr
        if hasattr(self.convs[0], "_edge_mask"):
            out_readout = self.readout(node_repr, batch, batch_size, edge_index=edge_index, edge_mask=self.convs[0]._edge_mask)
        else:
            out_readout = self.readout(node_repr, batch, batch_size)
        return out_readout

    def get_node_repr(self, x, edge_index, edge_attr, batch, batch_size, **kwargs):
        r"""
        The GIN encoder for molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            edge_attr (Tensor): edge attributes
            batch (Tensor): batch indicator
            batch_size (int): Batch size.

        Returns (Tensor):
            node feature representations
        """

        layer_feat = [self.atom_encoder(x)]

        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            tmp = conv(layer_feat[-1], edge_index, edge_attr)
            post_conv = batch_norm(tmp)
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            layer_feat.append(dropout(post_conv))
        return layer_feat[-1]


