r"""Training pipeline: training/evaluation structure, batch training.
"""
import datetime
import os
import shutil
from typing import Dict
from typing import Union
import random
import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_std, scatter_min, scatter_max, scatter_add
from munch import Munch
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data, InMemoryDataset
from torch_geometric.utils import to_networkx, from_networkx, to_undirected, sort_edge_index, shuffle_node, is_undirected, contains_self_loops, contains_isolated_nodes, coalesce, subgraph
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as sk_roc_auc, f1_score, accuracy_score
from imblearn.under_sampling import RandomUnderSampler

from GOOD.ood_algorithms.algorithms.BaseOOD import BaseOODAlg
from GOOD.utils.args import CommonArgs
from GOOD.utils.evaluation import eval_data_preprocess, eval_score
from GOOD.utils.logger import pbar_setting
from GOOD.utils.register import register
from GOOD.utils.train import nan2zero_get_mask
from GOOD.utils.initial import reset_random_seed
import GOOD.kernel.pipelines.xai_metric_utils as xai_utils
from GOOD.utils.splitting import split_graph, sparse_sort, relabel

import wandb

pbar_setting["disable"] = True

class CustomDataset(InMemoryDataset):
    def __init__(self, root, samples, belonging, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        data_list = []
        for i , G in enumerate(samples):
            if type(G) is nx.classes.digraph.DiGraph:
                data = from_networkx(G)
            else:
                data = copy.deepcopy(G)
                
            data.belonging = belonging[i]
            data.idx = i
            
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)


@register.pipeline_register
class Pipeline:
    r"""
    Kernel pipeline.

    Args:
        task (str): Current running task. 'train' or 'test'
        model (torch.nn.Module): The GNN model.
        loader (Union[DataLoader, Dict[str, DataLoader]]): The data loader.
        ood_algorithm (BaseOODAlg): The OOD algorithm.
        config (Union[CommonArgs, Munch]): Please refer to :ref:`configs:GOOD Configs and command line Arguments (CA)`.

    """

    def __init__(self, task: str, model: torch.nn.Module, loader: Union[DataLoader, Dict[str, DataLoader]],
                 ood_algorithm: BaseOODAlg,
                 config: Union[CommonArgs, Munch]):
        super(Pipeline, self).__init__()
        self.task: str = task
        self.model: torch.nn.Module = model
        self.loader: Union[DataLoader, Dict[str, DataLoader]] = loader
        self.ood_algorithm: BaseOODAlg = ood_algorithm
        self.config: Union[CommonArgs, Munch] = config

    def pretrain_batch_degenerate(self, loader: DataLoader, config: Union[CommonArgs, Munch]) -> dict:
        f1_pos_epoch, f1_neg_epoch, acc_epoch = 0, 0, 0
        epoch = -1
        while min(f1_pos_epoch, f1_neg_epoch, acc_epoch) < 0.99:
            epoch += 1
            self.config.train.epoch = epoch
            print(f'\nEpoch {epoch}:')

            per_batch_metrics = defaultdict(list)
            pbar = tqdm(enumerate(loader), total=len(loader), **pbar_setting)
            for index, data in pbar:
                
                # train a batch
                self.ood_algorithm.optimizer.zero_grad()
                data = data.to(self.config.device)

                mask, targets = nan2zero_get_mask(data, 'train', self.config)
                node_norm = data.get('node_norm') if self.config.model.model_level == 'node' else None
                data, _, mask, _ = self.ood_algorithm.input_preprocess(
                    data,
                    targets,
                    mask,
                    node_norm,
                    self.model.training,
                    self.config
                )

                model_output = self.model(
                    data=data,
                    edge_weight=None,
                    ood_algorithm=self.ood_algorithm,
                    max_num_epoch=self.config.train.max_epoch,
                    curr_epoch=epoch,
                    pretrain=True
                )

                # Train the classifier
                raw_pred = self.ood_algorithm.output_postprocess(model_output)       
                clf_loss = self.ood_algorithm.loss_calculate(raw_pred, targets, mask, node_norm, self.config, batch=data.batch).sum() / mask.sum()

                node_att = self.ood_algorithm.edge_att

                if len(data.y.shape) > 1:
                    graph_label_per_node = data.y.view(-1)[data.batch] # contains for each node the label of the graph it belongs to
                else:
                    graph_label_per_node = data.y[data.batch]

                targets = torch.zeros_like(data.node_is_spurious, dtype=torch.float, device=data.x.device)

                # blue_nodes_for_negative = torch.logical_and(data.x[:, 1] == 1, graph_label_per_node == 0).float()
                # red_nodes_for_positive = torch.logical_and(data.x[:, 0] == 1, graph_label_per_node == 1).float()
                # targets += blue_nodes_for_negative + red_nodes_for_positive

                # G iff R>=B; V iff R<B
                violet_nodes_for_negative = torch.logical_and(data.x[:, 3] == 1, graph_label_per_node == 0).float()
                green_nodes_for_positive = torch.logical_and(data.x[:, 2] == 1, graph_label_per_node == 1).float()                
                targets += green_nodes_for_positive + violet_nodes_for_negative

                # G+V iff R>=B; \emptyset iff R<B
                # violet_nodes_for_negative = torch.logical_and(data.x[:, 3] == 1, graph_label_per_node == 1).float()
                # green_nodes_for_positive = torch.logical_and(data.x[:, 2] == 1, graph_label_per_node == 1).float()                
                # targets += green_nodes_for_positive + violet_nodes_for_negative
                
                loss_weight = targets.clone()
                loss_weight[targets == 0] = 1
                loss_weight[targets == 1] = 10
                loss = F.binary_cross_entropy(node_att.squeeze(1), targets, weight=loss_weight)
                self.ood_algorithm.backward(loss + clf_loss)
                
                pred, target = eval_data_preprocess(data.y, raw_pred, mask, self.config)
                task_score = eval_score([pred], [target], self.config, pos_class=self.loader["train"].dataset.minority_class)
                f1_pos = f1_score(
                    targets.cpu().numpy(),
                    (node_att.squeeze(1) > 0.8).cpu().numpy(),
                    average='binary',
                    pos_label=1
                )
                f1_neg = f1_score(
                    targets.cpu().numpy(),
                    (node_att.squeeze(1) > 0.2).cpu().numpy(),
                    average='binary',
                    pos_label=0
                )
                per_batch_metrics["loss"].append(loss.item())
                per_batch_metrics["f1_pos"].append(f1_pos)
                per_batch_metrics["f1_neg"].append(f1_neg)
                f1_pos_epoch = np.mean(per_batch_metrics['f1_pos'])
                f1_neg_epoch = np.mean(per_batch_metrics['f1_neg'])
                acc_epoch = task_score
            
            print(
                f"Degenerate Loss: {np.mean(per_batch_metrics['loss']):.4f} " +
                f"F1_pos: {f1_pos_epoch:.2f} " +
                f"F1_neg: {f1_neg_epoch:.2f} " +
                f"Acc: {acc_epoch:.2f}"
            )

            # --- scheduler step ---
            self.ood_algorithm.scheduler.step()

        epoch_train_stat = self.evaluate(
            'eval_train',
            compute_wiou=False
        )
        id_val_stat = self.evaluate('id_val')
        id_test_stat = self.evaluate('id_test')
        val_stat = id_val_stat
        test_stat = id_test_stat
        loss_per_batch_dict = {}
        
        self.save_epoch(
            epoch,
            epoch_train_stat, id_val_stat, id_test_stat, val_stat, test_stat,
            self.config,
            loss_per_batch_dict,
            manual_save="pretrain_degenerate"
        )

        return None

    def train_batch(self, data: Batch, pbar, epoch:int) -> dict:
        r"""
        Train a batch. (Project use only)

        Args:
            data (Batch): Current batch of data.

        Returns:
            Calculated loss.
        """
        data = data.to(self.config.device)

        self.ood_algorithm.optimizer.zero_grad()

        mask, targets = nan2zero_get_mask(data, 'train', self.config)
        node_norm = data.get('node_norm') if self.config.model.model_level == 'node' else None
        data, targets, mask, node_norm = self.ood_algorithm.input_preprocess(data, targets, mask, node_norm,
                                                                             self.model.training,
                                                                             self.config)
        edge_weight = data.get('edge_norm') if self.config.model.model_level == 'node' else None

        model_output = self.model(
            data=data,
            edge_weight=edge_weight,
            ood_algorithm=self.ood_algorithm,
            max_num_epoch=self.config.train.max_epoch,
            curr_epoch=epoch
        )

        raw_pred = self.ood_algorithm.output_postprocess(model_output)
        
        loss = self.ood_algorithm.loss_calculate(raw_pred, targets, mask, node_norm, self.config, batch=data.batch)
        loss = self.ood_algorithm.loss_postprocess(loss, data, mask, self.config, epoch)

        # if self.config.dataset.dataset_name == "BAColorGVIsolated":
        #     loss += 0.01 * self.model.classifier.classifier[0].weight.abs().sum()

        self.ood_algorithm.backward(loss)
        
        pred, target = eval_data_preprocess(data.y, raw_pred, mask, self.config)

        return {
            'loss': loss.detach(),
            'score': eval_score([pred], [target], self.config, pos_class=self.loader["train"].dataset.minority_class), 
            'clf_loss': self.ood_algorithm.clf_loss,
            'l_norm_loss': self.ood_algorithm.l_norm_loss.item(),
            'entr_loss': self.ood_algorithm.entr_loss.item(),
            'spec_loss': self.ood_algorithm.spec_loss.item(),
            'mean_loss': self.ood_algorithm.mean_loss.item(),
            'total_loss': self.ood_algorithm.total_loss.item(),
        }


    def train(self):
        r"""
        Training pipeline.
        """
        if self.config.wandb:
            wandb.login()

        # config model
        print('Config model')
        self.config_model('train')

        # Load training utils
        print('Load training utils')
        self.ood_algorithm.set_up(self.model, self.config)

        print("Before training:")
        epoch_train_stat = self.evaluate('eval_train')
        id_val_stat = self.evaluate('id_val')
        id_test_stat = self.evaluate('id_test')

        if self.config.wandb:
            wandb.log({
                    "epoch": -1,
                    "all_train_loss": epoch_train_stat["loss"],
                    "all_id_val_loss": id_val_stat["loss"],
                    "train_score": epoch_train_stat["score"],
                    "id_val_score": id_val_stat["score"],
                    "id_test_score": id_test_stat["score"],
                    "val_score": np.nan,
                    "test_score": np.nan,
                },
                step=0
            )

        if self.config.train.pretrain_degenerate:
            # for param in self.model.classifierS.parameters():
            #     param.requires_grad = False

            print("#IM#Pretraining model for degenerate explanations")
            self.pretrain_batch_degenerate(self.loader['train'], self.config)
            print("#IM#End of pretraining")
            
            # for param in self.model.parameters():
            #     param.requires_grad = False
            
            # for param in self.model.classifierS.parameters():
            #     param.requires_grad = True

        # train the model
        counter = 1
        for epoch in range(self.config.train.ctn_epoch, self.config.train.max_epoch):
            self.config.train.epoch = epoch
            print(f'\nEpoch {epoch}:')

            self.ood_algorithm.stage_control(self.config)

            pbar = tqdm(enumerate(self.loader['train']), total=len(self.loader['train']), **pbar_setting)
            edge_scores = []
            train_batch_score = []
            loss_per_batch_dict = defaultdict(list)
            for index, data in pbar:
                if data.batch is not None and (data.batch[-1] < self.config.train.train_bs - 1):
                    continue

                # train a batch
                train_stat = self.train_batch(data, pbar, epoch)

                # log stats
                train_batch_score.append(train_stat["score"])
                for l in ("mean_loss", "spec_loss", "total_loss", "entr_loss", "l_norm_loss", "clf_loss"):
                    loss_per_batch_dict[l].append(train_stat.get(l, np.nan)) 

                if self.config.model.model_name != "GIN":
                    edge_scores.append(self.ood_algorithm.edge_att.detach().cpu())                                  
            
            for l in ("mean_loss", "spec_loss", "total_loss", "entr_loss", "l_norm_loss", "clf_loss"):
                loss_per_batch_dict[l] = np.mean(loss_per_batch_dict[l])

            # Epoch val
            print('Evaluating...')
            print(f"Clf loss: {loss_per_batch_dict['clf_loss']:.4f}")
            print(f"Spec loss: {loss_per_batch_dict['spec_loss']:.4f}")
            print(f"Mean loss: {loss_per_batch_dict['mean_loss']:.4f}")
            print(f"Total loss: {loss_per_batch_dict['total_loss']:.4f}")

            epoch_train_stat = self.evaluate(
                'eval_train',
                compute_wiou=False
            )
            id_val_stat = self.evaluate('id_val')
            id_test_stat = self.evaluate('id_test')
            
            if self.config.dataset.shift_type == "no_shift":
                val_stat = id_val_stat
                test_stat = id_test_stat
            else:
                # val_stat = self.evaluate('val')
                # test_stat = self.evaluate('test')
                val_stat = id_val_stat
                test_stat = id_test_stat

            if self.config.model.model_name != "GIN":
                tmp = torch.cat(edge_scores, dim=0)
                print("edge_weight: ", tmp.min(), tmp.max(), tmp.mean())

            if self.config.wandb:
                edge_scores = torch.cat(edge_scores, dim=0)
                log_dict = {
                    "epoch": epoch,
                    "clf_loss": loss_per_batch_dict["clf_loss"],
                    "mean_loss": loss_per_batch_dict["mean_loss"],
                    "spec_loss": loss_per_batch_dict["spec_loss"],
                    "total_loss": loss_per_batch_dict["total_loss"],
                    "l_norm_loss": loss_per_batch_dict["l_norm_loss"],
                    "entr_loss": loss_per_batch_dict["entr_loss"],
                    "all_train_loss": epoch_train_stat["loss"],
                    "all_id_val_loss": id_val_stat["loss"],
                    "train_batch_score": np.mean(train_batch_score),
                    "train_score": epoch_train_stat["score"],
                    "id_val_score": id_val_stat["score"],
                    "id_test_score": id_test_stat["score"],
                    "val_score": val_stat["score"],
                    "test_score": test_stat["score"],
                    "edge_weight": wandb.Histogram(sequence=edge_scores, num_bins=100),
                    "wiou": epoch_train_stat["wiou"],
                }
                wandb.log(log_dict, step=counter)
                counter += 1

            # checkpoints save
            self.save_epoch(epoch, epoch_train_stat, id_val_stat, id_test_stat, val_stat, test_stat, self.config, loss_per_batch_dict)

            # --- scheduler step ---
            self.ood_algorithm.scheduler.step()


    def plot_hist_score(self, data, density=False, log=False, name="noname.png"):        
        path = f'GOOD/kernel/pipelines/plots/attn_distrib/{self.config.load_split}_{self.config.dataset.dataset_name}_{self.config.dataset.domain}_{self.config.util_model_dirname}_{self.config.random_seed}/'            
        plt.hist(data, bins=100, density=density, log=log)
        plt.xlim(0.0,1.1)
        plt.title(f"distrib. edge_scores (min={round(min(data), 2)}, max={round(max(data), 2)})")
        plt.savefig(path + name)
        plt.close()
    

    @torch.no_grad()
    def get_subragphs_ratio(self, graphs, ratio, edge_scores, is_weight=False):
        """
            Cut graphs based on TopK or value thresholding strategy.
            If 'is_weight==False', use Top-'ratio'.
            Otherwise, use a thresholding with value 'ratio'
        """
        # # DEBUG OF SPARSE TOPK
        # i = 1
        # print("Weights:")
        # for j, (u,v) in enumerate(graphs[i].edge_index.T):
        #     if u < v:
        #         print((u.item(), v.item()), edge_scores[i][j])
        # print()

        # # case with twice the same graph (separated): Equal result        
        # print("\nSeparate and independent case (bs=1)\n")
        # (causal_edge_index, causal_edge_attr, causal_edge_weight), \
        #     (spu_edge_index, spu_edge_attr, spu_edge_weight) = split_graph(
        #         graphs[i],
        #         edge_scores[i],
        #         ratio
        #     )
        # print(causal_edge_index)
        # (causal_edge_index, causal_edge_attr, causal_edge_weight), \
        #     (spu_edge_index, spu_edge_attr, spu_edge_weight) = split_graph(
        #         graphs[i],
        #         edge_scores[i],
        #         ratio
        #     )
        # print(causal_edge_index)        
                
        # # case with twice the same graph (joined in batch): Equal result
        # print("\nJoined case (bs=3)\n")
        # data_joined = Batch().from_data_list([graphs[i], graphs[i], graphs[i]])

        # (causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch), \
        #     (spu_edge_index, spu_edge_attr, spu_edge_weight) = split_graph(
        #         data_joined,
        #         torch.cat((edge_scores[i], edge_scores[i], edge_scores[i]), dim=0),
        #         ratio,
        #         return_batch=True
        #     )

        # causal_edge_index = sort_edge_index(causal_edge_index)
        # for j, (u,v) in enumerate(causal_edge_index.T):
        #     num = graphs[i].num_nodes            
        #     if j % (len(causal_edge_index[0]) / 3) == 0:
        #         print("-"*20)
        #     u, v = int(u.item() - num * (j // (len(causal_edge_index[0]) / 3))) , int(v.item() - num * (j // (len(causal_edge_index[0]) / 3)))
        #     print((u, v), causal_edge_weight[j])
        # exit()
        # # END OF DEBUG
        

        spu_subgraphs, causal_subgraphs, expl_accs, causal_masks = [], [], [], []
        if "CIGA" in self.config.model.model_name:
            norm_edge_scores = [e.sigmoid() for e in edge_scores]
        else:
            norm_edge_scores = edge_scores

        # spu_subgraphs2, causal_subgraphs2, expl_accs2 = [], [], []
        # Select relevant subgraph (bs = 1)
        # for i in range(len(graphs)):
        #     (causal_edge_index, causal_edge_attr, causal_edge_weight), \
        #         (spu_edge_index, spu_edge_attr, spu_edge_weight) = split_graph(
        #             graphs[i],
        #             edge_scores[i],
        #             ratio
        #         )
        #     causal_subgraphs2.append(causal_edge_index.detach().cpu())
        #     spu_subgraphs2.append(spu_edge_index.detach().cpu())
        #     expl_accs2.append(xai_utils.expl_acc(causal_subgraphs2[-1], graphs[i], norm_edge_scores[i]) if hasattr(graphs[i], "edge_gt") else np.nan)

        # Select relevant subgraph (bs = all)
        big_data = Batch().from_data_list(graphs)        
        (causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch), \
            (spu_edge_index, spu_edge_attr, spu_edge_weight), mask = split_graph(
                big_data,
                torch.cat(edge_scores, dim=0),
                ratio,
                return_batch=True,
                is_weight=is_weight
            )
        
        cumnum = torch.tensor([g.num_nodes for g in graphs]).cumsum(0)
        cumnum[-1] = 0
        for j in range(causal_batch.max() + 1):
            causal_subgraphs.append(causal_edge_index[:, big_data.batch[causal_edge_index[0]] == j] - cumnum[j-1])
            spu_subgraphs.append(spu_edge_index[:, big_data.batch[spu_edge_index[0]] == j] - cumnum[j-1])
            expl_accs.append(xai_utils.expl_acc(causal_subgraphs[-1], graphs[j], norm_edge_scores[j]) if hasattr(graphs[j], "edge_gt") else (np.nan,np.nan))
            causal_masks.append(mask[big_data.batch[big_data.edge_index[0]] == j])


        # assert torch.allclose(torch.tensor(expl_accs), torch.tensor(expl_accs2), atol=1e-5)
        # for k in range(causal_batch.max()):
        #     assert torch.all(causal_subgraphs[k] == causal_subgraphs2[k]), f"\n{causal_subgraphs[k]}\n{causal_subgraphs2[k]}"
        #     assert torch.all(spu_subgraphs[k] == spu_subgraphs2[k]), f"\n{spu_subgraphs[k]}\n{spu_subgraphs2[k]}"

        # big_data = Batch().from_data_list(graphs[:10])        
        # (causal_edge_index2, causal_edge_attr, causal_edge_weight, causal_batch2), \
        #     (spu_edge_index, spu_edge_attr, spu_edge_weight) = split_graph(
        #         big_data,
        #         torch.cat(edge_scores[:10], dim=0),
        #         ratio,
        #         return_batch=True
        #     )            
        # print(edge_scores[0].dtype)
        # for k in range(5):
        #     print(sort_edge_index(causal_subgraphs[k]))
        #     print(sort_edge_index(causal_edge_index[:, causal_batch == k]) - sum([graphs[j].num_nodes for j in range(k)]))
        #     print(sort_edge_index(causal_edge_index2[:, causal_batch2 == k]) - sum([graphs[j].num_nodes for j in range(k)]))
        #     print("-"*20)
        # exit()
        
        return causal_subgraphs, spu_subgraphs, expl_accs, causal_masks
    
    @torch.no_grad()
    def get_subragphs_weight(self, graphs, weight, edge_scores):
        spu_subgraphs, causal_subgraphs, cau_idxs, spu_idxs = [], [], [], []
        expl_accs = []
        # Select relevant subgraph
        for i in range(len(graphs)):
            cau_idxs.append(edge_scores[i] >= weight)
            spu_idxs.append(edge_scores[i] < weight)

            spu = (graphs[i].edge_index.T[spu_idxs[-1]]).T
            cau = (graphs[i].edge_index.T[cau_idxs[-1]]).T

            causal_subgraphs.append(cau)
            spu_subgraphs.append(spu)
            expl_accs.append(xai_utils.expl_acc(cau, graphs[i]) if hasattr(graphs[i], "edge_gt") else np.nan)
        return causal_subgraphs, spu_subgraphs, expl_accs, cau_idxs, spu_idxs

    @torch.no_grad()
    def evaluate_graphs(self, loader, log=False, **kwargs):
        pbar = tqdm(loader, desc=f'Eval intervened graphs', total=len(loader), **pbar_setting)
        preds_eval, belonging = [], []
        for data in pbar:
            data: Batch = data.to(self.config.device)
            if log:
                output = self.model.log_probs(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm, **kwargs)
            else:
                output = self.model.probs(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm, **kwargs)
            preds_eval.extend(output.detach().cpu().numpy().tolist())
            belonging.extend(data.belonging.detach().cpu().numpy().tolist())
        preds_eval = torch.tensor(preds_eval)
        belonging = torch.tensor(belonging, dtype=int)
        return preds_eval, belonging

    # def get_intervened_graph(self, metric, graph, empty_idx=None, causal=None, spu=None, source=None, debug=None, idx=None, bank=None, feature_intervention=False, feature_bank=None):
    #     i, j, c = idx
    #     if metric == "fidm" or (metric == "suff" and causal is None):
    #         return xai_utils.sample_edges(graph, "spu", self.config.fidelity_alpha_2, spu)
    #     elif metric in ("nec", "nec++", "fidp"):
    #         if metric == "nec++":
    #             alpha = max(self.config.nec_alpha_1 - 0.1 * (j // 3), 0.1)
    #         else:
    #             alpha = self.config.nec_alpha_1
    #         return xai_utils.sample_edges(graph, alpha, deconfounded=True, edge_index_to_remove=causal)
    #     else:
    #         G_t = graph.copy()
    #         # xai_utils.mark_edges(G_t, causal, spu)
    #         G_t_filt = xai_utils.remove_from_graph(G_t, edge_index_to_remove=causal)
    #         num_elem = xai_utils.mark_frontier(G_t, G_t_filt)

    #         if len(G_t_filt) == 0:
    #             empty_idx.add(j)
    #             return None

    #         if feature_intervention:
    #             if i == 0 and j == 0:
    #                 print(f"Applying feature interventions with alpha = {self.config.feat_int_alpha}")
    #             G_t_filt = xai_utils.feature_intervention(G_t_filt, feature_bank, self.config.feat_int_alpha)

    #         G_union = xai_utils.random_attach_no_target_frontier(source, G_t_filt)
    #         if debug:
    #             if c <= 3 and i < 3:
    #                 xai_utils.draw(self.config, source, subfolder="plots_of_suff_scores", name=f"graph_{i}")
    #                 pos = xai_utils.draw(self.config, G_t, subfolder="plots_of_suff_scores", name=f"graph_{j}")
    #                 xai_utils.draw(self.config, G_t_filt, subfolder="plots_of_suff_scores", name=f"spu_graph_{j}", pos=pos)
    #                 xai_utils.draw(self.config, G_union, subfolder="plots_of_suff_scores", name=f"joined_graph_{i}_{j}")
    #             else:
    #                 exit()
    #     return G_union

    def get_indices_dataset(self, dataset, extract_all=False):
        if self.config.numsamples_budget == "all" or self.config.numsamples_budget >= len(dataset) or extract_all:
            idx = np.arange(len(dataset))        
        elif self.config.numsamples_budget < len(dataset):
            idx, _ = train_test_split(
                np.arange(len(dataset)),
                train_size=min(self.config.numsamples_budget, len(dataset)), # / len(dataset)
                random_state=42,
                shuffle=True,
                stratify=dataset.y if torch_geometric.__version__ == "2.4.0" else dataset.data.y
            )
        if "weight" in self.config.log_id or "class1" in self.config.log_id:
            if self.config.numsamples_budget < len(dataset):
                return train_test_split(
                    np.arange(len(dataset))[dataset.y.reshape(-1) == 1],
                    train_size=min(self.config.numsamples_budget, len(dataset)), # / len(dataset)
                    random_state=42,
                    shuffle=True,
                )[0]
            else:
                return np.arange(len(dataset))[dataset.y.reshape(-1) == 1] # TODO: remove this
        return idx

    @torch.no_grad()
    def generate_binary_explanations(self, is_weight, thrs, splits, convert_to_nx, is_node_expl):
        assert is_node_expl
        
        reset_random_seed(self.config)
        self.model.eval()

        samples = {
            split: 
                {thr: [] for thr in thrs} 
            for split in splits
        }
        avg_graph_size = {}
        graphs_nx = {}
        for split in splits:
            dataset = self.get_local_dataset(split)
            loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=2)            
            for data in loader:
                data: Batch = data.to(self.config.device)   
                edge_scores, node_scores, logits = self.model.get_subgraph(
                    data=data,
                    edge_weight=None,
                    ood_algorithm=self.ood_algorithm,
                    do_relabel=False
                )

                for j, g in enumerate(data.to_data_list()):
                    for thr in thrs:
                        new_g = copy.deepcopy(g)

                        if is_node_expl:
                            node_expl = node_scores[data.batch == j].squeeze(1)
                            new_g.node_expl = node_expl

                            # compute binary node mask based on threshold here for convenience
                            new_g.node_mask = new_g.node_expl >= thr
                            
                            # compute binary edge mask from previous node mask
                            # take the node induced subgraph as topological explanation
                            _, _, new_g.edge_mask = subgraph(new_g.node_mask, new_g.edge_index, None, return_edge_mask=True)
                        else:
                            raise ValueError("only node expl for now")
                        
                        samples[split][thr].append(new_g)
            
            avg_graph_size[split] = np.mean([g.edge_index.shape[1] for g in samples[split][thrs[0]]])

            if convert_to_nx:
                print("Converting graphs to networkx")
                edge_attr_tokeep = [s for s in ["edge_attr", "edge_gt", "edge_attr"] if s in g.keys()]
                graphs_nx[split] = [to_networkx(g, node_attrs=["x", "node_is_spurious"], edge_attrs=edge_attr_tokeep or None) for g in samples[split][thrs[0]]]
            else:
                graphs_nx[split] = list()
        return samples, graphs_nx, avg_graph_size


    @torch.no_grad()
    def compute_metric(
        self,
        metric: str,
        graphs,
        graphs_nx,
        avg_graph_size,
    ):
        assert metric in ["suff", "fidm", "nec", "nec++", "fidp", "suff++", "suff_simple", "interven_suff"]

        print(f"\n\n", "-"*50)
        reset_random_seed(self.config)
        self.model.eval()   

        scores, acc_ints = defaultdict(list), []

        eval_samples, belonging, reference = [], [], []
        preds_ori, labels_ori, expl_acc_ori = [], [], []
        empty_idx = set()

        pbar = tqdm(range(len(graphs)), desc=f'Creating Intervent. distrib.', total=len(graphs), **pbar_setting)
        for i in pbar:
            # if metric in ("suff", "suff++"):
            #     G = graphs_nx[i].copy()
            #     G_filt = xai_utils.remove_from_graph(G, edge_index_to_remove=spu_subgraphs_r[ratio][i])
            #     num_elem = xai_utils.mark_frontier(G, G_filt)
            #     if len(G_filt) == 0 or num_elem == 0:
            #         continue

            # if metric in ("fidm", "fidp", "nec", "nec++") or len(empty_idx) == len(graphs):
            #     intervened_graphs = xai_utils.sample_edges_tensorized_batched(
            #         graphs[i],
            #         nec_number_samples=self.config.nec_number_samples,
            #         nec_alpha_1=self.config.nec_alpha_1,
            #         avg_graph_size=avg_graph_size,
            #         edge_index_to_remove=causal_masks_r[ratio][i],
            #         sampling_type=self.config.samplingtype,
            #         budget=self.config.expval_budget
            #     )
            # elif metric == "suff" or metric == "suff++" or metric == "suff_simple":
            #     if ratio == 1.0:
            #         eval_samples.extend([graphs[i]]*self.config.expval_budget)
            #         belonging.extend([i]*self.config.expval_budget)
            #     else:
            #         z, c = -1, 0
            #         idxs = np.random.permutation(np.arange(len(labels))) #pick random from every class
            #         budget = self.config.expval_budget
                    
            #         if metric == "suff++":
            #             budget = budget // 2
            #         if metric == "suff_simple":
            #             budget = 0 # skip interventions and just pick subsamples

            #         while c < budget:
            #             if z == len(idxs) - 1:
            #                 break
            #             z += 1
            #             j = idxs[z]
            #             if j in empty_idx:
            #                 continue

            #             G_union = self.get_intervened_graph(
            #                 metric,
            #                 graphs_nx[j],
            #                 empty_idx,
            #                 causal_subgraphs_r[ratio][j],
            #                 spu_subgraphs_r[ratio][j],
            #                 G_filt,
            #                 debug,
            #                 (i, j, c),
            #                 feature_intervention=False,
            #                 feature_bank=None
            #             )
            #             if G_union is None:
            #                 continue
            #             eval_samples.append(G_union)
            #             belonging.append(i)
            #             c += 1
                    
            #         intervened_graphs = xai_utils.sample_edges_tensorized_batched(
            #             graphs[i],
            #             nec_number_samples=self.config.nec_number_samples,
            #             nec_alpha_1=self.config.nec_alpha_1*2,
            #             avg_graph_size=avg_graph_size,
            #             edge_index_to_remove=~causal_masks_r[ratio][i],
            #             sampling_type=self.config.samplingtype,
            #             budget=self.config.expval_budget
            #         )

            if metric == "fidm" or metric == "fidp":
                intervened_graphs = xai_utils.fidelity(
                    graphs[i],
                    type=metric
                )

            if intervened_graphs is not None:
                eval_samples.append(graphs[i])
                reference.append(len(eval_samples) - 1)
                belonging.append(-1)
                labels_ori.append(graphs[i].y)
                belonging.extend([i] * len(intervened_graphs))
                eval_samples.extend(intervened_graphs)

                # idx = 322
                # if reference[-1] == idx:
                #     print(graphs[i])
                #     print(graphs[i].edge_index)
                #     print(intervened_graphs[0].edge_index)

                #     for name, g in zip(["ori", "pertb"], [graphs[i], intervened_graphs[0]]):
                #         G = to_networkx(g, node_attrs=["x", "node_expl"], to_undirected=True)
                #         xai_utils.draw_colored(
                #             self.config,
                #             G,
                #             node_expl=g.node_expl,
                #             subfolder=f"debug_metrics/{self.config.ood_dirname}/{self.config.dataset.dataset_name}_{self.config.dataset.domain}",
                #             name=f"{name}_{idx}",
                #             thrs=0.5,
                #             title=f"Idx: {i} Class={int(g.y.item())}",
                #             with_labels=True,
                #             figsize=(6.4, 4.8)
                #         )
                #     exit()

        if len(eval_samples) <= 1:
            print(f"\nToo few intervened samples, skipping this")
            exit()
            for c in labels_ori_ori.unique():
                scores[c.item()].append(1.0)
            scores["all_KL"].append(1.0)
            scores["all_L1"].append(1.0)
            acc_ints.append(-1.0)
            return None
        
        ##
        # Compute new predictions
        ##
        int_dataset = CustomDataset(root=None, samples=eval_samples, belonging=belonging)
        loader = DataLoader(int_dataset, batch_size=256, shuffle=False)
        preds_eval, belonging = self.evaluate_graphs(loader, log=False)

        preds_clean_graphs = preds_eval[reference]
        
        mask = torch.ones(preds_eval.shape[0], dtype=bool)
        mask[reference] = False
        preds_perturbed_graphs = preds_eval[mask]
        belonging = belonging[mask]            
        assert torch.all(belonging >= 0), f"{torch.all(belonging >= 0)}"

        num_perturbation_per_sample = 1 if metric in ("fidm", "fidp") else self.config.expval_budget
        labels_ori = torch.tensor(labels_ori)
        preds_clean_graphs = preds_clean_graphs.repeat_interleave(num_perturbation_per_sample, dim=0)
        labels_ori_repeated = labels_ori.repeat_interleave(num_perturbation_per_sample, dim=0)

        ##
        # Debug predictions and explanations
        ##
        # cls = 0
        # print(
        #     torch.cat(
        #         (
        #             labels_ori_repeated.unsqueeze(1)[labels_ori_repeated==cls],
        #             preds_clean_graphs[labels_ori_repeated==cls],
        #             preds_perturbed_graphs[labels_ori_repeated==cls]
        #         ), 
        #         dim=1
        #     )
        # )
        # for idx in torch.nonzero(labels_ori == cls):
        #     g = eval_samples[reference[idx]]
        #     print(idx, belonging[idx], reference[idx], g.y.item(), preds_clean_graphs[idx], preds_perturbed_graphs[idx])
        #     print(g.x.sum(0), g.x[g.node_mask].sum(0))
        #     print(
        #         eval_samples[reference[idx]+1].x.sum(0),
        #         self.model.probs(data=Batch.from_data_list([eval_samples[reference[idx]+1]]), edge_weight=None, ood_algorithm=self.ood_algorithm).item()
        #     )
            # Print new explanation scores
            # if idx in [267, 268]:
            #     edge_scores, node_scores, logits = self.model.get_subgraph(
            #         data=Batch.from_data_list([eval_samples[reference[idx]+1]]),
            #         edge_weight=None,
            #         ood_algorithm=self.ood_algorithm,
            #         do_relabel=False,
            #         return_attn=False,
            #         ratio=None
            #     )
            #     print("Explanation on pertub sample: ", node_scores.view(-1), node_scores.view(-1).max().item(), node_scores.view(-1).mean().item())
            #     print(logits)
            # Plot explanations
            # if idx in [154, 130, 161]:
            #     print(eval_samples[reference[idx]+1].edge_index)
            #     g = eval_samples[reference[idx]+1]
            #     G = to_networkx(g, node_attrs=["x", "node_expl"], to_undirected=True)
            #     xai_utils.draw_colored(
            #         self.config,
            #         G,
            #         node_expl=g.node_expl,
            #         subfolder=f"debug_metrics/{self.config.ood_dirname}/{self.config.dataset.dataset_name}_{self.config.dataset.domain}",
            #         name=f"debug_fid_{idx}",
            #         thrs=0.5,
            #         title=f"Idx: {i} Class={int(g.y.item())} Pred={preds_perturbed_graphs[idx].item():.2f}",
            #         with_labels=False,
            #         figsize=(6.4, 4.8)
            #     )
            # print()

        ##
        # Compute metric values
        ##
        aggr = self.get_aggregated_metric(
            metric,
            preds_clean_graphs,
            preds_perturbed_graphs,
            belonging
        )
        
        ##
        # Store and print metric values
        ##
        for m in ["TV", "predicted"]:
            for c in labels_ori.unique():
                idx_class = np.arange(labels_ori.shape[0])[(labels_ori == c).numpy()]
                scores[f"{c.item()}_{m}"].append(round(aggr[m][idx_class].mean().item(), 3))
            scores[f"all_{m}"].append(round(aggr[m].mean().item(), 3))

        acc_clean = eval_score(preds_clean_graphs, labels_ori_repeated, self.config, self.loader["id_val"].dataset.minority_class)
        acc_interven = eval_score(preds_perturbed_graphs, labels_ori_repeated, self.config, self.loader["id_val"].dataset.minority_class)
        acc_ints.append(acc_interven.item())

        print()
        print(f"Label distrib: {labels_ori.unique(return_counts=True)}")
        print(f"Acc clean", round(acc_clean.item(), 3))
        print(f"Acc interven", round(acc_interven.item(), 3))
        print(f"len(reference) = {len(reference)}")
        for m in ["TV", "predicted"]:
            for c in labels_ori.unique().numpy().tolist():
                print(f"{metric.upper()} for class {c}_{m} = {scores[str(c)+'_'+m][-1]} +- {aggr['predicted'].std():.3f} (in-sample avg dev_std = {(aggr['std_predicted']**2).mean().sqrt():.3f})")
            print(f"{metric.upper()} all_classes {m} = {scores[f'all_{m}'][-1]} +- {aggr[f'{m}'].std():.3f} (in-sample avg dev_std =", torch.round((aggr[f"std_{m}"]**2).mean().sqrt(), decimals=3).item())
        return scores, acc_ints


    def normalize_belonging(self, belonging):
        #TODO: make more efficient
        ret = []
        i = -1
        for j , elem in enumerate(belonging):
            if len(ret) > 0 and elem == belonging[j-1]:
                ret.append(i)
            else:
                i += 1
                ret.append(i)
        return ret

    def get_aggregated_metric(self, metric, preds_clean, preds_perturb, belonging):
        ret = {}
        belonging = torch.tensor(self.normalize_belonging(belonging))

        div_TV = torch.abs(preds_clean - preds_perturb).sum(-1)
        ret["TV"] = scatter_mean(div_TV, belonging, dim=0) # average across perturbations
        if preds_clean.shape[1] == 1:
            div_predicted = torch.abs(preds_clean - preds_perturb)
        else:
            exit("Check if .gather() is working")
            pred_class = preds_clean.argmax(-1)
            div_predicted = torch.abs(
                preds_clean.gather(1, pred_class) - preds_perturb.gather(1, pred_class)
            )

        ret["predicted"] = scatter_mean(div_predicted, belonging, dim=0) # average across perturbations
        ret["std_TV"] = scatter_std(div_TV, belonging, dim=0)
        ret["std_predicted"] = scatter_std(div_predicted, belonging, dim=0)
        return ret

    def get_local_dataset(self, split, log=True):
        if torch_geometric.__version__ == "2.4.0" and log:
            print(self.loader[split].dataset, "for split ", split)
            print(f"Data example from {split}: {self.loader[split].dataset.get(0)}")
            print(f"Label distribution from {split}: {self.loader[split].dataset.y.unique(return_counts=True)}")        

        dataset = self.loader[split].dataset
        
        if abs(dataset.y.unique(return_counts=True)[1].min() - dataset.y.unique(return_counts=True)[1].max()) > 1000:
            print(f"#D#Unbalanced warning for {self.config.dataset.dataset_name} ({split})")
        
        if "hiv" in self.config.dataset.dataset_name.lower() and str(self.config.numsamples_budget) != "all":
            balanced_idx, _ = RandomUnderSampler(random_state=42).fit_resample(np.arange(len(dataset)).reshape(-1,1), dataset.y)

            dataset = dataset[balanced_idx.reshape(-1)]
            print(f"Creating balanced dataset: {dataset.y.unique(return_counts=True)}")
        return dataset

    @torch.no_grad()
    def evaluate(self, split: str, compute_suff=False, compute_wiou=False, compute_clf_only_pred=False):
        r"""
        This function is design to collect data results and calculate scores and loss given a dataset subset.
        (For project use only)

        Args:
            split (str): A split string for choosing the corresponding dataloader. Allowed: 'train', 'id_val', 'id_test',
                'val', and 'test'.

        Returns:
            A score and a loss.

        """
        stat = {'score': None, 'loss': None, 'wiou': None}
        if self.loader.get(split) is None:
            return stat
        
        was_training = self.model.training
        self.model.eval()

        # self.model.gnn.encoder.batch_norms.train()
        # for conv in self.model.gnn.encoder.convs:
        #     conv.mlp.train()
        # self.model.gnn.eval()

        loss_all = []
        mask_all = []
        pred_all = []
        pred_clf_only_all = []
        target_all = []
        likelihoods_all = []
        wious_all = []
        pbar = tqdm(self.loader[split], desc=f'Eval {split.capitalize()}', total=len(self.loader[split]),
                    **pbar_setting)
        c = 0
        for data in pbar:
            data: Batch = data.to(self.config.device)

            # c += 1
            # if c < 2:
            #     continue
            

            mask, targets = nan2zero_get_mask(data, split, self.config)
            if mask is None:
                return stat
            node_norm = torch.ones((data.num_nodes,),
                                   device=self.config.device) if self.config.model.model_level == 'node' else None
            data, targets, mask, node_norm = self.ood_algorithm.input_preprocess(data, targets, mask, node_norm,
                                                                                 self.model.training,
                                                                                 self.config)
            model_output = self.model(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm)
            raw_preds = self.ood_algorithm.output_postprocess(model_output)

            if compute_clf_only_pred:
                clf_only_output = self.model.predict_from_subgraph(
                    data=data,
                    edge_att=torch.ones((data.edge_index.shape[1]), device=data.x.device),
                    node_att=torch.ones((data.x.shape[0],1), device=data.x.device)
                ).squeeze(-1)
                pred_clf_only_all.append(clf_only_output.cpu().numpy())

            # --------------- Loss collection ------------------
            loss: torch.tensor = self.config.metric.loss_func(raw_preds, targets, reduction='none') * mask
            mask_all.append(mask)
            loss_all.append(loss)

            # ------------- Likelihood data collection ------------------
            if raw_preds.shape[-1] > 1:
                probs = raw_preds.softmax(dim=1)
                likelihoods_all.append(probs.gather(1, targets.unsqueeze(1)))
            else:
                probs = raw_preds.sigmoid()
                likelihoods_all.append(torch.full_like(probs, fill_value=-1))            

            # ------------- Score data collection ------------------
            pred, target = eval_data_preprocess(data.y, raw_preds, mask, self.config)
            pred_all.append(pred)
            target_all.append(target)

            # ------------- WIOU ------------------
            if compute_wiou:
                wious_mask = torch.ones(data.batch.max() + 1, dtype=torch.bool)
                if self.config.dataset.dataset_name == "TopoFeature" or self.config.dataset.dataset_name == "SimpleMotif":
                    wious_mask[data.pattern == 0] = False # Mask out examples without the motif
                
                _, explanation = to_undirected(data.edge_index, self.ood_algorithm.edge_att.squeeze(-1), reduce="mean")
                wious_all.append(
                    xai_utils.expl_acc_super_fast(data, explanation, reference_intersection=data.edge_gt)[wious_mask].mean().item()
                )

        # ------- Loss calculate -------
        loss_all = torch.cat(loss_all)
        mask_all = torch.cat(mask_all)
        likelihoods_all = torch.cat(likelihoods_all)
        stat['loss'] = loss_all.sum() / mask_all.sum()
        stat['likelihood_avg'] = likelihoods_all.mean()
        stat['likelihood_prod'] = torch.prod(likelihoods_all)
        stat['likelihood_logprod'] = torch.sum(likelihoods_all.log())
        stat['wiou'] = np.mean(wious_all) if len(wious_all) > 0 else np.nan

        # --------------- Metric calculation including ROC_AUC, Accuracy, AP.  --------------------
        stat['score'] = eval_score(pred_all, target_all, self.config, self.loader[split].dataset.minority_class)

        print(
            f'{split.capitalize()} {self.config.metric.score_name}: {stat["score"]:.4f} \t' + 
            f'{split.capitalize()} Loss: {stat["loss"]:.4f} \t' + 
            (f'{split.capitalize()} WIoU: {stat["wiou"]:.3f} \t' if compute_wiou else '')
        )

        if was_training:
            self.model.train()

        return {
            'score': stat['score'],
            'loss': stat['loss'],
            'likelihood_avg': stat['likelihood_avg'],
            'likelihood_prod': stat['likelihood_prod'],
            'likelihood_logprod': stat['likelihood_logprod'],
            'wiou': stat['wiou'],
            'pred': pred_all,
            'pred_clf_only': pred_clf_only_all
        }

    def load_task(self, load_param=False, load_split="ood"):
        r"""
        Launch a training or a test.
        """
        if self.task == 'train':
            self.train()
            return None, None
        elif self.task == 'test':
            # config model
            print('#D#Config model and output the best checkpoint info...')
            test_score, ckpt = self.config_model('test', load_param=load_param, load_split=load_split)
            return test_score, ckpt

    def config_model(self, mode: str, load_param=False, load_split="ood"):
        r"""
        A model configuration utility. Responsible for transiting model from CPU -> GPU and loading checkpoints.
        Args:
            mode (str): 'train' or 'test'.
            load_param: When True, loading test checkpoint will load parameters to the GNN model.

        Returns:
            Test score and loss if mode=='test'.
        """
        self.model.to(self.config.device)
        self.model.train()

        # load checkpoint
        if mode == 'train' and self.config.train.tr_ctn:
            assert False
            ckpt = torch.load(os.path.join(self.config.ckpt_dir, f'last.ckpt'))
            self.model.load_state_dict(ckpt['state_dict'])
            best_ckpt = torch.load(os.path.join(self.config.ckpt_dir, f'best.ckpt'))
            self.config.metric.best_stat['score'] = best_ckpt['val_score']
            self.config.metric.best_stat['loss'] = best_ckpt['val_loss']
            self.config.train.ctn_epoch = ckpt['epoch'] + 1
            print(f'#IN#Continue training from Epoch {ckpt["epoch"]}...')

        if mode == 'test':
            try:
                ckpt = torch.load(self.config.test_ckpt, map_location=self.config.device)
            except FileNotFoundError:
                print(f'#E#Checkpoint not found at {os.path.abspath(self.config.test_ckpt)}')
                exit(1)
            if os.path.exists(self.config.id_test_ckpt):
                id_ckpt = torch.load(self.config.id_test_ckpt, map_location=self.config.device)
                # model.load_state_dict(id_ckpt['state_dict'])
                print(f'#IN#Loading best In-Domain Checkpoint {id_ckpt["epoch"]} in {self.config.id_test_ckpt}')
                print(f'#IN#Checkpoint {id_ckpt["epoch"]}: \n-----------------------------------\n'
                      f'Train {self.config.metric.score_name}: {id_ckpt["train_score"]:.4f}\n'
                      f'Train Loss: {id_ckpt.get("train_loss", np.nan):.4f}\n'
                      f'Spec Loss: {id_ckpt.get("spec_loss", np.nan):.4f}\n'
                      f'Mean Loss: {id_ckpt.get("mean_loss", np.nan):.4f}\n'
                      f'Total Loss: {id_ckpt.get("total_loss", np.nan):.4f}\n'
                      f'ID Validation {self.config.metric.score_name}: {id_ckpt["id_val_score"]:.4f}\n'
                      f'ID Validation Loss: {id_ckpt["id_val_loss"].item():.4f}\n'
                      f'ID Test {self.config.metric.score_name}: {id_ckpt["id_test_score"]:.4f}\n'
                      f'ID Test Loss: {id_ckpt["id_test_loss"].item():.4f}\n'
                      f'OOD Validation {self.config.metric.score_name}: {id_ckpt["val_score"]:.4f}\n'
                      f'OOD Validation Loss: {id_ckpt["val_loss"].item():.4f}\n'
                      f'OOD Test {self.config.metric.score_name}: {id_ckpt["test_score"]:.4f}\n'
                      f'OOD Test Loss: {id_ckpt["test_loss"].item():.4f}\n')
                print(f'#IN#Loading best Out-of-Domain Checkpoint {ckpt["epoch"]}...')
                print(f'#IN#Checkpoint {ckpt["epoch"]}: \n-----------------------------------\n'
                      f'Train {self.config.metric.score_name}: {ckpt["train_score"]:.4f}\n'
                      f'Train Loss: {ckpt["train_loss"].item():.4f}\n'
                      f'ID Validation {self.config.metric.score_name}: {ckpt["id_val_score"]:.4f}\n'
                      f'ID Validation Loss: {ckpt["id_val_loss"].item():.4f}\n'
                      f'ID Test {self.config.metric.score_name}: {ckpt["id_test_score"]:.4f}\n'
                      f'ID Test Loss: {ckpt["id_test_loss"].item():.4f}\n'
                      f'OOD Validation {self.config.metric.score_name}: {ckpt["val_score"]:.4f}\n'
                      f'OOD Validation Loss: {ckpt["val_loss"].item():.4f}\n'
                      f'OOD Test {self.config.metric.score_name}: {ckpt["test_score"]:.4f}\n'
                      f'OOD Test Loss: {ckpt["test_loss"].item():.4f}\n')

                print(f'#IN#ChartInfo {id_ckpt["id_test_score"]:.4f} {id_ckpt["test_score"]:.4f} '
                      f'{ckpt["id_test_score"]:.4f} {ckpt["test_score"]:.4f} {ckpt["id_val_score"]:.4f} {ckpt["val_score"]:.4f}', end='')

            else:
                print(f'#IN#No In-Domain checkpoint.')
                # model.load_state_dict(ckpt['state_dict'])
                print(f'#IN#Loading best Checkpoint {ckpt["epoch"]}...')
                print(f'#IN#Checkpoint {ckpt["epoch"]}: \n-----------------------------------\n'
                      f'Train {self.config.metric.score_name}: {ckpt["train_score"]:.4f}\n'
                      f'Train Loss: {ckpt["train_loss"].item():.4f}\n'
                      f'Validation {self.config.metric.score_name}: {ckpt["val_score"]:.4f}\n'
                      f'Validation Loss: {ckpt["val_loss"].item():.4f}\n'
                      f'Test {self.config.metric.score_name}: {ckpt["test_score"]:.4f}\n'
                      f'Test Loss: {ckpt["test_loss"].item():.4f}\n')

                print(
                    f'#IN#ChartInfo {ckpt["test_score"]:.4f} {ckpt["val_score"]:.4f}', end='')
            if load_param:
                if self.config.ood.ood_alg != 'EERM':
                    if load_split == "ood":
                        self.model.load_state_dict(ckpt['state_dict'])
                    elif load_split == "id":
                        self.model.load_state_dict(id_ckpt['state_dict'])
                    else:
                        raise ValueError(f"{load_split} not supported")
                else:
                    self.model.gnn.load_state_dict(ckpt['state_dict'])
            return ckpt["test_score"], id_ckpt

    # @torch.no_grad()
    # def save_epoch(self, epoch: int, train_stat: dir, id_val_stat: dir, id_test_stat: dir, val_stat: dir,
    #                test_stat: dir, config: Union[CommonArgs, Munch]):
    #     r"""
    #     Training util for checkpoint saving.

    #     Args:
    #         epoch (int): epoch number
    #         train_stat (dir): train statistics
    #         id_val_stat (dir): in-domain validation statistics
    #         id_test_stat (dir): in-domain test statistics
    #         val_stat (dir): ood validation statistics
    #         test_stat (dir): ood test statistics
    #         config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.ckpt_dir`, :obj:`config.dataset`, :obj:`config.train`, :obj:`config.model`, :obj:`config.metric`, :obj:`config.log_path`, :obj:`config.ood`)

    #     Returns:
    #         None

    #     """
    #     if epoch < config.train.pre_train:
    #         return

    #     if not (config.metric.best_stat['score'] is None or config.metric.lower_better * val_stat[
    #         'score'] < config.metric.lower_better *
    #             config.metric.best_stat['score']
    #             or (id_val_stat.get('score') and (
    #                     config.metric.id_best_stat['score'] is None or config.metric.lower_better * id_val_stat[
    #                 'score'] < config.metric.lower_better * config.metric.id_best_stat['score']))
    #             or epoch % config.train.save_gap == 0):
    #         return
        
    #     state_dict = self.model.state_dict() if config.ood.ood_alg != 'EERM' else self.model.gnn.state_dict()
    #     ckpt = {
    #         'state_dict': state_dict,
    #         'train_score': train_stat['score'],
    #         'train_loss': train_stat['loss'],
    #         'id_val_score': id_val_stat['score'],
    #         'id_val_loss': id_val_stat['loss'],
    #         'id_test_score': id_test_stat['score'],
    #         'id_test_loss': id_test_stat['loss'],
    #         'val_score': val_stat['score'],
    #         'val_loss': val_stat['loss'],
    #         'test_score': test_stat['score'],
    #         'test_loss': test_stat['loss'],
    #         'time': datetime.datetime.now().strftime('%b%d %Hh %M:%S'),
    #         'model': {
    #             'model name': f'{config.model.model_name} {config.model.model_level} layers',
    #             'dim_hidden': config.model.dim_hidden,
    #             'dim_ffn': config.model.dim_ffn,
    #             'global pooling': config.model.global_pool
    #         },
    #         'dataset': config.dataset.dataset_name,
    #         'train': {
    #             'weight_decay': config.train.weight_decay,
    #             'learning_rate': config.train.lr,
    #             'mile stone': config.train.mile_stones,
    #             'shift_type': config.dataset.shift_type,
    #             'Batch size': f'{config.train.train_bs}, {config.train.val_bs}, {config.train.test_bs}'
    #         },
    #         'OOD': {
    #             'OOD alg': config.ood.ood_alg,
    #             'OOD param': config.ood.ood_param,
    #             'number of environments': config.dataset.num_envs
    #         },
    #         'log file': config.log_path,
    #         'epoch': epoch,
    #         'max epoch': config.train.max_epoch
    #     }

    #     if not os.path.exists(config.ckpt_dir):
    #         os.makedirs(config.ckpt_dir)
    #         print(f'#W#Directory does not exists. Have built it automatically.\n'
    #               f'{os.path.abspath(config.ckpt_dir)}')

    #     saved_file = os.path.join(config.ckpt_dir, f'last.ckpt')
    #     torch.save(ckpt, saved_file)

    #     if not config.clean_save:
    #         # WARNING: Original code was saving every epoch and then if 'clean_save' delete checkpoint
    #         shutil.copy(saved_file, os.path.join(config.ckpt_dir, f'{epoch}.ckpt'))

    #     # --- In-Domain checkpoint ---
    #     # WARNING: Original code saves if 'score' is not None AND if valiation score is grater than best validation score
    #     # if id_val_stat.get('score') and (
    #     #         config.metric.id_best_stat['score'] is None or config.metric.lower_better * id_val_stat[
    #     #     'score'] < config.metric.lower_better * config.metric.id_best_stat['score']):
    #     if id_val_stat.get('loss') and (
    #             config.metric.id_best_stat['loss'] is None or id_val_stat['loss'] < config.metric.id_best_stat['loss']):            
    #         config.metric.id_best_stat['score'] = id_val_stat['score']
    #         config.metric.id_best_stat['loss'] = id_val_stat['loss']
    #         shutil.copy(saved_file, os.path.join(config.ckpt_dir, f'id_best.ckpt'))
    #         print('#IM#Saved a new best In-Domain checkpoint based on validation loss.')

    #     # --- Out-Of-Domain checkpoint ---
    #     # if id_val_stat.get('score'):
    #     #     if not (config.metric.lower_better * id_val_stat['score'] < config.metric.lower_better * val_stat['score']):
    #     #         return
    #     if val_stat.get('loss') and (
    #             config.metric.best_stat['loss'] is None or val_stat['loss'] < config.metric.best_stat['loss']):
            
    #         config.metric.best_stat['score'] = val_stat['score']
    #         config.metric.best_stat['loss'] = val_stat['loss']
    #         shutil.copy(saved_file, os.path.join(config.ckpt_dir, f'best.ckpt'))
    #         print('#IM#Saved a new best OOD checkpoint based on validation loss.')

    def save_epoch(self, epoch: int, train_stat: dir, id_val_stat: dir, id_test_stat: dir, val_stat: dir,
                   test_stat: dir, config: Union[CommonArgs, Munch], loss_per_batch_dict: dict, manual_save:str=None):
        r"""
        Training util for checkpoint saving.

        Args:
            epoch (int): epoch number
            train_stat (dir): train statistics
            id_val_stat (dir): in-domain validation statistics
            id_test_stat (dir): in-domain test statistics
            val_stat (dir): ood validation statistics
            test_stat (dir): ood test statistics
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.ckpt_dir`, :obj:`config.dataset`, :obj:`config.train`, :obj:`config.model`, :obj:`config.metric`, :obj:`config.log_path`, :obj:`config.ood`)

        Returns:
            None

        """
        state_dict = self.model.state_dict() if config.ood.ood_alg != 'EERM' else self.model.gnn.state_dict()
        ckpt = {
            'state_dict': state_dict,
            'train_score': train_stat['score'],
            'train_loss': train_stat['loss'],
            'id_val_score': id_val_stat['score'],
            'id_val_loss': id_val_stat['loss'],
            'id_test_score': id_test_stat['score'],
            'id_test_loss': id_test_stat['loss'],
            'val_score': val_stat['score'],
            'val_loss': val_stat['loss'],
            'test_score': test_stat['score'],
            'test_loss': test_stat['loss'],
            'time': datetime.datetime.now().strftime('%b%d %Hh %M:%S'),
            'model': {
                'model name': f'{config.model.model_name} {config.model.model_level} layers',
                'dim_hidden': config.model.dim_hidden,
                'dim_ffn': config.model.dim_ffn,
                'global pooling': config.model.global_pool
            },
            'dataset': config.dataset.dataset_name,
            'train': {
                'weight_decay': config.train.weight_decay,
                'learning_rate': config.train.lr,
                'mile stone': config.train.mile_stones,
                'shift_type': config.dataset.shift_type,
                'Batch size': f'{config.train.train_bs}, {config.train.val_bs}, {config.train.test_bs}'
            },
            'OOD': {
                'OOD alg': config.ood.ood_alg,
                'OOD param': config.ood.ood_param,
                'number of environments': config.dataset.num_envs
            },
            'log file': config.log_path,
            'epoch': epoch,
            'max epoch': config.train.max_epoch
        }
        ckpt.update(loss_per_batch_dict)

        if epoch < config.train.pre_train:
            return

        # WARNING: Original reference metric is 'score'
        reference_metric = "loss"
        lower_better = 1 if reference_metric == "loss" else -1

        if not (config.metric.best_stat[reference_metric] is None or 
                lower_better * val_stat[reference_metric] < lower_better *
                config.metric.best_stat[reference_metric]
            or (id_val_stat.get(reference_metric) and (
                        config.metric.id_best_stat[reference_metric] is None or 
                        lower_better * id_val_stat[reference_metric] < lower_better * config.metric.id_best_stat[reference_metric]))
            or epoch % config.train.save_gap == 0):
            return

        if not os.path.exists(config.ckpt_dir):
            os.makedirs(config.ckpt_dir)
            print(f'#W#Directory does not exists. Have built it automatically.\n'
                  f'{os.path.abspath(config.ckpt_dir)}')
        
        saved_file = os.path.join(config.ckpt_dir, f'{epoch}.ckpt')
        torch.save(ckpt, saved_file)
        shutil.copy(saved_file, os.path.join(config.ckpt_dir, f'last.ckpt'))

        if manual_save is not None:
            print(f'#W#Saving manual checkpoint {manual_save}.ckpt')
            shutil.copy(saved_file, os.path.join(config.ckpt_dir, f'{manual_save}.ckpt'))

        # --- In-Domain checkpoint ---
        if id_val_stat.get(reference_metric) and (
                config.metric.id_best_stat[reference_metric] is None or lower_better * id_val_stat[
            reference_metric] < lower_better * config.metric.id_best_stat[reference_metric]):
            config.metric.id_best_stat['score'] = id_val_stat['score']
            config.metric.id_best_stat['loss'] = id_val_stat['loss']
            shutil.copy(saved_file, os.path.join(config.ckpt_dir, f'id_best.ckpt'))
            print('#IM#Saved a new best In-Domain checkpoint.')

        # --- Out-Of-Domain checkpoint ---
        if config.metric.best_stat[reference_metric] is None or lower_better * val_stat[
            reference_metric] < lower_better * \
                config.metric.best_stat[reference_metric]:
            config.metric.best_stat['score'] = val_stat['score']
            config.metric.best_stat['loss'] = val_stat['loss']
            shutil.copy(saved_file, os.path.join(config.ckpt_dir, f'best.ckpt'))
            print('#IM#Saved a new best checkpoint.')
        
        if config.clean_save:
            os.unlink(saved_file)

    def get_node_explanations(self):
        self.model.eval()
        
        # self.model.gnn.encoder.batch_norms.train()
        # for conv in self.model.gnn.encoder.convs:
        #     conv.mlp.train() # Make BN of ConvLayer in train mode

        splits = ["id_val"]
        ret = {
            split: {
                "scores": [],
                "samples": [],
                "pred": []
            } for split in splits
        }
                
        for i, split in enumerate(splits):
            dataset = self.get_local_dataset(split)

            loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=2)
            for data in loader:
                data: Batch = data.to(self.config.device)   
                edge_scores, node_scores, logits = self.model.get_subgraph(
                    data=data,
                    edge_weight=None,
                    ood_algorithm=self.ood_algorithm,
                    do_relabel=False
                )

                for j, g in enumerate(data.to_data_list()):
                    node_expl = node_scores[data.batch == j].detach().cpu().numpy().squeeze(1)
                    ret[split]["scores"].append(node_expl.tolist())
                    ret[split]["samples"].append(g)
                    ret[split]["pred"].append(logits[j])
        return ret

    def generate_panel(self):
        self.model.eval()
        
        # self.model.gnn.encoder.batch_norms.train()
        # for conv in self.model.gnn.encoder.convs:
        #     conv.mlp.train() # Make BN of ConvLayer in train mode

        splits = ["train", "id_val", "id_test"] #, "test"
        n_row = 1
        fig, axs = plt.subplots(n_row, len(splits), figsize=(9,4))
        # plt.suptitle(f"{self.config.model.model_name[:4]}") # - {self.config.dataset.dataset_name} {self.config.dataset.domain}
        
        for i, split in enumerate(splits):            
            # acc = self.evaluate(split, compute_suff=False)["score"]
            # print(f"Acc ({split}) =  ({acc:.3f}%)")
            dataset = self.get_local_dataset(split)

            loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=2)
            edge_scores, effective_ratios = [], []
            for data in loader:
                data: Batch = data.to(self.config.device)   
                edge_score = self.model.get_subgraph(
                                data=data,
                                edge_weight=None,
                                ood_algorithm=self.ood_algorithm,
                                do_relabel=False
                        )
                for j, g in enumerate(data.to_data_list()):
                    edge_scores.append(edge_score[data.batch[data.edge_index[0]] == j].detach().cpu().numpy().tolist())
                    if g.edge_index.shape[1] > 0:
                        effective_ratios.append(float((g.edge_gt.sum() if hasattr(g, "edge_gt") and not g.edge_gt is None else 0.) / (g.edge_index.shape[1])))
            
            if "CIGA" in self.config.model.model_name:
                edge_scores = [np.abs(np.array(e)) for e in edge_scores]
                edge_scores = [(e - e.min()) / (e.max() - e.min() + 1e-7) for e in edge_scores if len(e) > 0]

            return edge_scores

            print(min(np.concatenate(edge_scores)), max(np.concatenate(edge_scores)), np.mean(np.concatenate(edge_scores)), np.std(np.concatenate(edge_scores)))
            axs[int(i/n_row)].hist(np.concatenate(edge_scores) + np.random.normal(0, 0.001, np.concatenate(edge_scores).shape), density=True, log=False, bins=100) #100 or np.linspace(-1, 1, 100)
            # a,b = np.unique(np.concatenate(edge_scores), return_counts=True)
            # print(a)
            # print(b)
            # axs[int(i/n_row)].bar(a,b)
            # axs[int(i/n_row)].set_title(f"{split}")
            # axs[int(i/n_row)].set_xlabel(f"explanation scores")
            # axs[int(i/n_row)].set_ylabel(f"density")
            axs[int(i/n_row)].set_xlim(-0.1, 1.1)  #axs[int(i/n_row)]
            axs[int(i/n_row)].set_ylim(0.0, 100)

            fig.supxlabel('explanation relevance scores', fontsize=13)
            fig.supylabel('density', fontsize=13)

            # means, stds = zip(*[(np.mean(e), np.std(e)) for e in edge_scores])
            # means, stds = self.smooth(np.array(means), k=5), np.array(stds)
            # axs[n_row - int(i/n_row) -1, n_row - int(i%n_row) -1].plot(np.arange(len(means)), means)
            # axs[n_row - int(i/n_row) -1, n_row - int(i%n_row) -1].plot(np.arange(len(effective_ratios)), self.smooth(effective_ratios, k=7), 'r', alpha=0.7)
            # axs[n_row - int(i/n_row) -1, n_row - int(i%n_row) -1].fill_between(np.arange(len(means)), means - stds, means + stds, alpha=0.5)
            # axs[n_row - int(i/n_row) -1, n_row - int(i%n_row) -1].set_title(f"Per sample attn. variability - {split}")
            # axs[n_row - int(i/n_row) -1, n_row - int(i%n_row) -1].set_ylim(0, 1.1)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        path = f'GOOD/kernel/pipelines/plots/panels/{self.config.ood_dirname}/'
        if not os.path.exists(path):
            os.makedirs(path)

        path += f"{self.config.load_split}_{self.config.dataset.dataset_name}_{self.config.dataset.domain}_{self.config.util_model_dirname}_{self.config.random_seed}"
        plt.savefig(path + ".png")
        plt.savefig(f'GOOD/kernel/pipelines/plots/panels/pdfs/{self.config.load_split}_{self.config.dataset.dataset_name}_{self.config.dataset.domain}_{self.config.util_model_dirname}_{self.config.random_seed}.pdf')
        print("\n Saved plot ", path, "\n")
        plt.close()
        return edge_scores
    
    def generate_panel_all_seeds(self, edge_scores_seed):
        n_row, n_col = 5, 2
        fig, axs = plt.subplots(n_row, n_col, figsize=(20,16))
        for j in range(len(edge_scores_seed)):   
            ax = axs[j // n_col, j % n_col]

            ax.hist(np.concatenate(edge_scores_seed[j]), density=True, log=False, bins=100)
            ax.set_title(f"seed {j+1}", fontsize=15)
            ax.set_yticks([])
            ax.tick_params(axis='y', labelsize=12)
            ax.tick_params(axis='x', labelsize=12)
            ax.set_xlim((0.0,1.0))
        fig.supxlabel('explanation relevance scores', fontsize=18)
        fig.supylabel('density', fontsize=18)
        fig.suptitle(self.config.model.model_name.replace("GIN", ""), fontsize=22)
        
        path = f'GOOD/kernel/pipelines/plots/panels/{self.config.ood_dirname}/'
        if not os.path.exists(path):
            os.makedirs(path)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        path += f"{self.config.load_split}_{self.config.dataset.dataset_name}_{self.config.dataset.domain}_{self.config.util_model_dirname}_allseeds"
        plt.savefig(path + ".png")
        plt.savefig(f'GOOD/kernel/pipelines/plots/panels/pdfs_new/{self.config.load_split}_{self.config.dataset.dataset_name}_{self.config.dataset.domain}_{self.config.util_model_dirname}_allseeds.pdf')
        print("\n Saved plot ", path, "\n")
        plt.close()

    def smooth(self, y, k):
        box = np.ones(k) / k
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth


    def generate_plot_sampling_type(self, splits, ratios, sampling_alphas, graphs, graphs_nx, causal_subgraphs_r, causal_masks, avg_graph_size):
        def nec_kl(ori_pred, pred, belonging, labels):
            return 1 - torch.exp(-scatter_mean(torch.nn.KLDivLoss(reduction="none", log_target=True)(ori_pred, pred).sum(-1), belonging, dim=0))
        def nec_l1(ori_pred, pred, belonging, labels):
            return 1 - torch.exp(-scatter_mean(torch.abs(ori_pred.exp() - pred.exp()).sum(-1), belonging, dim=0))
        def fid_l1_div(ori_pred, pred, belonging, labels):
            return torch.abs(ori_pred.exp() - pred.exp()).sum(-1).mean()
        def fid_model(ori_pred, pred, belonging, labels):
            return torch.abs(ori_pred.exp().gather(1, ori_pred.argmax(-1).unsqueeze(1)) - pred.exp().gather(1, ori_pred.argmax(-1).unsqueeze(1))).mean()
        def fid_phenom(ori_pred, pred, belonging, labels):
            return torch.abs(ori_pred.exp().gather(1, labels.long().unsqueeze(1)) - pred.exp().gather(1, labels.long().unsqueeze(1))).mean()
        def ratio_pred_change(ori_pred, pred, belonging, labels):
            return sum(ori_pred.argmax(-1) != pred.argmax(-1)) / pred.shape[0]
        def predict_sample(graphs, ratio):
            eval_set = CustomDataset("", graphs, torch.arange(len(graphs)))
            loader = DataLoader(eval_set, batch_size=256, shuffle=False, num_workers=0)
            preds, belonging = self.evaluate_graphs(loader, log=True, weight=ratio, is_ratio=True, eval_kl=True)
            return preds
    
        self.model.eval()

        metrics, accs = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
        for SPLIT in splits:
            for ratio in ratios:
                ori_graphs = []
                G_sampled_rfid, G_sampled_deconf, G_sampled_fixed, G_sampled_deconf_R = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
                belonging, labels = [], torch.tensor([])
                labels_norep = torch.tensor([])
                for idx in range(len(graphs[SPLIT])):
                    # reset_random_seed(self.config)
                    # if graphs[SPLIT][idx].edge_index.shape[1] <= 3 or graphs[SPLIT][idx].num_nodes <= 3:
                    #     continue
                    labels_norep = torch.cat((labels_norep, graphs[SPLIT][idx].y), dim=0)
                    for j in range(self.config.expval_budget):
                        for alpha in sampling_alphas:
                            sample = xai_utils.sample_edges_tensorized(graphs[SPLIT][idx], nec_number_samples=None, nec_alpha_1=alpha, avg_graph_size=None, sampling_type="bernoulli", edge_index_to_remove=causal_masks[SPLIT][ratio][idx], force_undirected=True)
                            G_sampled_rfid[alpha].append(sample)
                        for alpha in sampling_alphas:
                            sample = xai_utils.sample_edges_tensorized(graphs[SPLIT][idx], nec_number_samples="prop_G_dataset", nec_alpha_1=alpha, avg_graph_size=avg_graph_size[SPLIT], sampling_type="deconfounded", edge_index_to_remove=causal_masks[SPLIT][ratio][idx], force_undirected=True)
                            G_sampled_deconf[alpha].append(sample)
                        for alpha in sampling_alphas:
                            sample = xai_utils.sample_edges_tensorized(graphs[SPLIT][idx], nec_number_samples="prop_R", nec_alpha_1=alpha, avg_graph_size=avg_graph_size[SPLIT], sampling_type="deconfounded", edge_index_to_remove=causal_masks[SPLIT][ratio][idx], force_undirected=True)
                            G_sampled_deconf_R[alpha].append(sample)
                        # for k in range(1, len(sampling_alphas)+1):
                        #     sample = xai_utils.sample_edges_tensorized(graphs[SPLIT][idx], nec_number_samples="alwaysK", nec_alpha_1=k, avg_graph_size=None, sampling_type="deconfounded", edge_index_to_remove=causal_masks[SPLIT][ratio][idx], force_undirected=True)
                        #     G_sampled_fixed[k].append(sample)

                        ori_graphs.append(graphs[SPLIT][idx])
                        belonging.append(idx)
                        labels = torch.cat((labels, graphs[SPLIT][idx].y), dim=0)

                labels = labels.view(-1)                
                belonging = torch.tensor(self.normalize_belonging(belonging), dtype=torch.long, device=labels.device)
                ori_pred = predict_sample(ori_graphs, ratio)
                divergences = {}                

                for k, alpha in enumerate(sampling_alphas):
                    k = k + 1 #for G_sampled_fixed
                    divergences[f"RFID_{alpha}"] = {}
                    divergences[f"DECONF_{alpha}"] = {}
                    divergences[f"DECONF_R_{alpha}"] = {}
                    # divergences[f"FIXED_{k}"] = {}
                    preds = predict_sample(G_sampled_rfid[alpha] + G_sampled_deconf[alpha] + G_sampled_deconf_R[alpha], ratio)
                    pred_rfid   = preds[:len(G_sampled_rfid[alpha])]
                    pred_deconf = preds[len(G_sampled_rfid[alpha]): len(G_sampled_rfid[alpha]) + len(G_sampled_deconf[alpha])]
                    pred_deconf_R = preds[len(G_sampled_rfid[alpha]) + len(G_sampled_deconf[alpha]):]
                    

                    # pred_fixed  = preds[len(G_sampled_rfid[alpha]) + len(G_sampled_deconf[alpha]):]
                    for metric_name, div_f in zip(["NEC L1"], [nec_kl, nec_l1]): #"FID L1 div", fid_l1_div
                        divergences[f"RFID_{alpha}"][metric_name]   = div_f(ori_pred, pred_rfid, belonging, labels).mean().item()
                        divergences[f"DECONF_{alpha}"][metric_name] = div_f(ori_pred, pred_deconf, belonging, labels).mean().item()
                        divergences[f"DECONF_R_{alpha}"][metric_name] = div_f(ori_pred, pred_deconf_R, belonging, labels).mean().item()
                        # divergences[f"FIXED_{k}"][metric_name]      = div_f(ori_pred, pred_fixed, belonging, labels).mean().item()
                        
                metrics[SPLIT][ratio] = divergences
                accs[SPLIT][ratio] = sum(labels == ori_pred.argmax(-1)) / ori_pred.shape[0]

                # pred_all = self.predict_sample(ori_graphs + [s[0] for s in sampled] + [s[1] for s in sampled] + [s[2] for s in sampled] + [s[3] for s in sampled] + [s[4] for s in sampled], tested_pipeline, ratio=ratio)
                # pred = pred_all[len(ori_graphs):len(ori_graphs) + len(sampled)]
                # pred_rfid = pred_all[len(ori_graphs) + len(sampled):len(ori_graphs) + 2*len(sampled)]
                # ori_preds[SPLIT][ratio] = ori_pred
                # preds[SPLIT][ratio] = pred
                # for metric_name, div_f in zip(["NEC KL", "NEC L1", "FID L1 div", "Model FID", "Phen. FID", "Change pred"], [nec_kl, nec_l1, fid_l1_div, fid_model, fid_phenom, ratio_pred_change]):
                #     divergences["alphaAvgG"][metric_name]        = div_f(ori_pred, pred, belonging, labels).mean().item()
        return metrics, accs


    @torch.no_grad()
    def generate_explanation_examples(
        self,
        seed,
        ratios,
        split: str,
        metric: str,
        edge_scores,
        graphs,
        graphs_nx,
        labels,
        avg_graph_size,
        causal_subgraphs_r,
        spu_subgraphs_r,
        expl_accs_r,
        causal_masks_r,
        intervention_bank,
        intervention_distrib:str = "model_dependent",
        debug=False,
    ):
        print(f"\n\n")
        print("-"*50)
        print(f"\n\n#D#Plotting examples of explanations for {metric.upper()} over {split} across ratios (random_expl={self.config.random_expl})")
        reset_random_seed(self.config)
        self.model.eval()   

        scores, results, acc_ints = defaultdict(list), {}, []
        ratios = [1.0]
        for ratio in ratios:
            reset_random_seed(self.config)
            print(f"\n\nratio={ratio}\n\n")            

            eval_samples, belonging, reference = [], [], []
            preds_ori, labels_ori, expl_acc_ori = [], [], []
            empty_idx = set()

            pbar = tqdm(range(len(edge_scores)), desc=f'Creating Intervent. distrib.', total=len(edge_scores), **pbar_setting)
            for i in pbar:
                if graphs[i].edge_index.shape[1] <= 6:
                    continue                
                if metric in ("suff", "suff++") and intervention_distrib == "model_dependent":
                    G = graphs_nx[i].copy()
                    G_filt = xai_utils.remove_from_graph(G, edge_index_to_remove=spu_subgraphs_r[ratio][i])
                    num_elem = xai_utils.mark_frontier(G, G_filt)
                    if len(G_filt) == 0 or num_elem == 0:
                        continue
                
                eval_samples.append(graphs[i])
                reference.append(len(eval_samples) - 1)
                belonging.append(-1)
                labels_ori.append(labels[i])
                # expl_acc_ori.append(expl_accs_r[ratio][i])

                continue # Added before ICML to just plot clean samples

                if metric in ("nec", "nec++") or len(empty_idx) == len(graphs) or intervention_distrib in ("fixed", "bank"):
                    assert False, "Computing NEC interventions"
                    if metric in ("suff", "suff++", "suff_simple") and intervention_distrib in ("fixed", "bank") and i == 0:
                        print(f"Using {intervention_distrib} interventional distribution")
                    elif metric in ("suff", "suff++", "suff_simple") and intervention_distrib == "model_dependent":
                        pass

                    for m in range(self.config.expval_budget):                        
                        G_c = xai_utils.sample_edges_tensorized(
                            graphs[i],
                            nec_number_samples=self.config.nec_number_samples,
                            nec_alpha_1=self.config.nec_alpha_1,
                            avg_graph_size=avg_graph_size,
                            edge_index_to_remove=causal_masks_r[ratio][i],
                            sampling_type="bernoulli" if metric in ("fidm", "fidp") else self.config.samplingtype
                        )
                        belonging.append(i)
                        eval_samples.append(G_c)
                elif metric == "suff" or metric == "suff++" or metric == "suff_simple":
                    if ratio == 1.0:
                        eval_samples.extend([graphs[i]]*self.config.expval_budget)
                        belonging.extend([i]*self.config.expval_budget)
                    else:
                        z, c = -1, 0
                        idxs = np.random.permutation(np.arange(len(labels))) #pick random from every class
                        budget = self.config.expval_budget
                        if metric == "suff++":
                            budget = budget // 2
                        if metric == "suff_simple":
                            budget = 0 # just pick subsamples
                        while c < budget:
                            assert False, "I'm using suff-Simple at the moment"
                            if z == len(idxs) - 1:
                                break
                            z += 1
                            j = idxs[z]
                            if j in empty_idx:
                                continue

                            G_union = self.get_intervened_graph(
                                metric,
                                intervention_distrib,
                                graphs_nx[j],
                                empty_idx,
                                causal_subgraphs_r[ratio][j],
                                spu_subgraphs_r[ratio][j],
                                G_filt,
                                debug,
                                (i, j, c),
                                feature_intervention=False,
                                feature_bank=None
                            )
                            if G_union is None:
                                continue
                            eval_samples.append(G_union)
                            belonging.append(i)
                            c += 1
                        for k in range(c, self.config.expval_budget): # if not enough interventions, pad with sub-sampling
                            G_c = xai_utils.sample_edges_tensorized(
                                graphs[i],
                                nec_number_samples=self.config.nec_number_samples,
                                nec_alpha_1=self.config.nec_alpha_1,
                                avg_graph_size=avg_graph_size,
                                edge_index_to_remove=~graphs[i].edge_gt, #~causal_masks_r[ratio][i]&
                                sampling_type="bernoulli" if metric in ("fidm", "fidp") else self.config.samplingtype
                            )
                            belonging.append(i)
                            eval_samples.append(G_c)
            
            int_dataset = CustomDataset("", eval_samples, belonging)
            loader = DataLoader(int_dataset, batch_size=512, shuffle=False)

            # thrs = 0.05
            if self.config.model.model_name == "GSATGIN":
                if self.config.dataset.dataset_name == "TopoFeature":
                    thrs = 0.8
                if self.config.dataset.dataset_name == "AIDSC1":
                    thrs = 0.7
                if self.config.dataset.dataset_name == "AIDS":
                    thrs = 0.8
                if self.config.dataset.dataset_name == "BAColor":
                    thrs = 0.7
            elif self.config.model.model_name == "SMGNNGIN":
                if self.config.dataset.dataset_name == "TopoFeature":
                    if self.config.global_side_channel == "simple_concept2temperature":
                        thrs = 0.20
                if self.config.dataset.dataset_name == "BAColor":
                    thrs = 0.2

            # PLOT EXAMPLES OF EXPLANATIONS (ORIGINAL SAMPLES)
            print(len(edge_scores), len(graphs), len(int_dataset), self.config.expval_budget, avg_graph_size)
            for i in range(25):
                if i > 25:
                    break
                data = int_dataset[reference[i]]
                # g = to_networkx(data, node_attrs=["node_gt", "x"], to_undirected=True) # when node_gt present
                g = to_networkx(data, node_attrs=["x"], to_undirected=True)
                # xai_utils.mark_edges(g, data.edge_index, data.edge_index[:, data.edge_gt == 1], inv_edge_w=edge_scores[i])
                xai_utils.mark_edges(g, data.edge_index, torch.tensor([[]]), inv_edge_w=edge_scores[i])
                xai_utils.draw_colored(
                    self.config,
                    g,
                    subfolder=f"plots_of_explanation_examples/{self.config.ood_dirname}/{self.config.dataset.dataset_name}_{self.config.dataset.domain}",
                    name=f"graph_{reference[i]}",
                    thrs=thrs,
                    title=f"Idx: {i} Class {labels_ori[reference[i]].long().item()}",
                    with_labels=False,
                    figsize=(12,10) if "AIDS" in self.config.dataset.dataset_name else (6.4, 4.8)
                )
                print(f"graph_{reference[i]} is of class {labels_ori[reference[i]]}")

            # PLOT EXAMPLES OF EXPLANATIONS (INTERVENED SAMPLES)
            # for i, data in enumerate(loader):
            #     if i > 15:
            #         break
            #     data: Batch = data.to(self.config.device)
            #     edge_score = self.model.get_subgraph(
            #         data=data,
            #         edge_weight=None,
            #         ood_algorithm=self.ood_algorithm,
            #         do_relabel=False,
            #         return_attn=False,
            #         ratio=None
            #     )
            #     wiou = xai_utils.expl_acc_super_fast(None, data, edge_score)[0]
            #     g = to_networkx(data, node_attrs=["node_gt"], to_undirected=True) #to_undirected=True
            #     xai_utils.mark_edges(g, data.edge_index, data.edge_index[:, data.edge_gt == 1], inv_edge_w=edge_score)
            #     xai_utils.draw(self.config, g, subfolder="plots_of_gt_stability_det", name=f"expl_{i}", title=wiou)
        return 
    
