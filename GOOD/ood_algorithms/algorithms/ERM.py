"""
Implementation of the baseline ERM
"""
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg

from torch import Tensor
from torch_geometric.data import Batch


@register.ood_alg_register
class ERM(BaseOODAlg):
    r"""
    Implementation of the baseline ERM

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(ERM, self).__init__(config)
        
        self.spec_loss = -1
        self.entr_loss = -1 
        self.l_norm_loss = -1
        self.clf_loss = -1

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], epoch:int,
                         **kwargs) -> Tensor:
        self.mean_loss = loss.mean()
        self.total_loss = self.mean_loss
        return self.total_loss
