r"""
Applies a linear transformation to complete classification from representations.
"""
import torch
import torch.nn as nn
from torch import Tensor

from GOOD.utils.config_reader import Union, CommonArgs, Munch


class Classifier(torch.nn.Module):
    r"""
    Applies a linear transformation to complete classification from representations.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.dataset.num_classes`)
    """
    def __init__(self, config: Union[CommonArgs, Munch], output_dim:int=None, is_linear:bool=True):

        super(Classifier, self).__init__()
        

        if config.model.gnn_clf_layer == 0 and config.mitigation_sampling == "raw":
            hidden_dim = config.dataset.dim_node
        else:
            hidden_dim = config.model.dim_hidden

        if is_linear:
            self.classifier = nn.Sequential(*(
                [
                    nn.Linear(
                        hidden_dim,
                        config.dataset.num_classes if output_dim is None else output_dim,
                        bias=False
                    )
                ]
            ))
        else:
            print("Using non linear ClassifierS")
            self.classifier = nn.Sequential(*(
                [
                    nn.Linear(hidden_dim, config.model.dim_hidden, bias=False),
                    torch.nn.LeakyReLU(),
                    nn.Linear(hidden_dim, config.model.dim_hidden, bias=False),
                    torch.nn.LeakyReLU(),
                    nn.Linear(hidden_dim, config.dataset.num_classes if output_dim is None else output_dim, bias=False)
                ]
            ))

    def forward(self, feat: Tensor) -> Tensor:
        r"""
        Applies a linear transformation to feature representations.

        Args:
            feat (Tensor): feature representations

        Returns (Tensor):
            label predictions

        """
        return self.classifier(feat)
