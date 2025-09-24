"""
The GOOD-SST2 dataset. Adapted from `DIG <https://github.com/divelab/DIG>`_.
"""
import itertools
import os
import os.path as osp
import random
from copy import deepcopy
from collections import defaultdict

import gdown
import numpy as np
import torch
from dig.xgraph.dataset import SentiGraphDataset
from munch import Munch
from torch_geometric.data import InMemoryDataset, extract_zip, Data
from torch_geometric.utils import subgraph
from torch.utils.data import random_split
from GOOD import register


@register.dataset_register
class GraphSST2Planted(InMemoryDataset):
    r"""
        Variant of Graph-SST2 dataset where we inplant a single ',' and a single '.' in every sample.
        ',' is added at the beginning, '.' at the end.
    """

    def __init__(self, root: str, subset: str = 'train', transform=None, pre_transform=None):

        self.name = self.__class__.__name__
        self.minority_class = None
        self.metric = 'Accuracy'
        self.task = 'Binary classification'
        self.domain = "basis"
        self.shift = "no_shift"

        self.comma_embed = torch.zeros((1, 768))
        self.period_embed = torch.ones((1, 768))

        super().__init__(root, transform, pre_transform)

        shift_mode = {'no_shift': 0, 'covariate': 3, 'concept': 8}
        mode = {'train': 0, 'id_val': 1, 'id_test': 2, 'val': 3, 'test': 4}
        subset_pt = shift_mode[self.shift] + mode[subset]

        self.data, self.slices = torch.load(self.processed_paths[subset_pt])

    @property
    def raw_dir(self):
        return osp.join(self.root)

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, self.domain, 'processed')

    @property
    def processed_file_names(self):
        return ['no_shift_train.pt', 'no_shift_val.pt', 'no_shift_test.pt']
    
    def process(self):
        dataset = SentiGraphDataset(root=self.root, name='GraphSST2')
        print('Load data done!')
        print('Num graphs = ', dataset.data.y.shape, "; shape of X = ", dataset.data.x.shape)

        dataset.data.y = dataset.data.y.unsqueeze(1).float()

        train_list = []
        test_list = []
        for i, data in enumerate(dataset):
            # Applying the same filter as DIR paper
            degree = float(data.num_edges) / data.num_nodes
            if data.num_edges <= 2:
                continue

            data.idx = i
            sentence_tokens = np.array(dataset.supplement['sentence_tokens'][str(i)])

            # remove ',' and '.' already present
            remove_idx = np.where((sentence_tokens == ",") | (sentence_tokens == "."))[0]
            if remove_idx.size > 0:

                to_keep = np.ones_like(sentence_tokens, dtype=bool)
                to_keep[remove_idx] = False
                sentence_tokens = sentence_tokens[to_keep]
                
                data.x = data.x[to_keep, :]
                data.edge_index, _ = subgraph(
                    subset=torch.tensor(to_keep),
                    edge_index=data.edge_index,
                    relabel_nodes=True,
                    num_nodes=data.x.shape[0]
                )

            # add new uncorrelated ',' and '.' (isolated nodes)
            data.x = torch.cat(
                (data.x, self.comma_embed, self.period_embed),
                dim=0
            )
            
            data.node_type = torch.zeros(data.x.shape[0])
            data.node_type[-1] = 1
            data.node_type[-2] = 2

            data.num_nodes = data.x.shape[0]
            sentence_tokens = np.concatenate((sentence_tokens, [','], ['.']))
            data.sentence_tokens = sentence_tokens.tolist()

            if degree >= 1.76785714:
                train_list.append(data)
            elif degree <= 1.57142857:
                test_list.append(data)
        self.num_data = train_list.__len__() + test_list.__len__()
        
        print('Extract data done!')

        val_list = train_list[:int(len(train_list) * 0.1)]
        train_list = train_list[int(len(train_list) * 0.1):]

        all_data_list = [train_list, val_list, test_list]
        for i, final_data_list in enumerate(all_data_list):
            data, slices = self.collate(final_data_list)
            torch.save((data, slices), self.processed_paths[i])

    @staticmethod
    def load(dataset_root: str, domain: str, shift: str = 'no_shift', generate: bool = False, debias: bool = False, model_name:str=None, add_pos_feat=None):
        r"""
        A staticmethod for dataset loading. This method instantiates dataset class, constructing train, id_val, id_test,
        ood_val (val), and ood_test (test) splits. Besides, it collects several dataset meta information for further
        utilization.

        Args:
            dataset_root (str): The dataset saving root.
            domain (str): The domain selection. Allowed: 'degree' and 'time'.
            shift (str): The distributional shift we pick. Allowed: 'no_shift', 'covariate', and 'concept'.
            generate (bool): The flag for regenerating dataset. True: regenerate. False: download.

        Returns:
            dataset or dataset splits.
            dataset meta info.
        """
        assert domain == "basis" and shift == "no_shift", f"domain = {domain}, shift = {shift}"

        meta_info = Munch()
        meta_info.dataset_type = 'nlp'
        meta_info.model_level = 'graph'        

        train_dataset = GraphSST2Planted(root=dataset_root, subset='train')
        id_val_dataset = GraphSST2Planted(root=dataset_root, subset='id_val')
        id_test_dataset = GraphSST2Planted(root=dataset_root, subset='id_test')

        # Are bringing/movie/written/career a nice degenerate explanation to present?
        # Idxs: 19, 24, 33, 36
        # count = defaultdict(int)
        # for d in train_dataset:            
        #     found = True
        #     for s in ["in"]:
        #         if s not in d.sentence_tokens:
        #             found = False
        #     if found:
        #         print(d.y.item(), " ".join(d.sentence_tokens))
        #         count[d.y.item()] += 1
        # print("\n", count)
        # exit("bringing")


        if "DIR" in model_name:
            train_dataset._data.y = train_dataset._data.y.squeeze(-1).long()
            id_val_dataset._data.y = id_val_dataset._data.y.squeeze(-1).long()
            id_test_dataset._data.y = id_test_dataset._data.y.squeeze(-1).long()

        val_dataset = id_val_dataset
        test_dataset = id_test_dataset

        meta_info.dim_node = train_dataset.num_node_features
        meta_info.dim_edge = train_dataset.num_edge_features
        meta_info.num_envs = 0

        # Define networks' output shape.
        if "DIR" in model_name:
            print("The task in 'Multi-label classification'")
            meta_info.num_classes = torch.unique(train_dataset._data.y).shape[0]
            task = 'Multi-label classification'
        else:
            task = 'Binary classification'
            meta_info.num_classes = 1

        # --- clear buffer dataset._data_list ---
        train_dataset._data_list = None
        if id_val_dataset:
            id_val_dataset._data_list = None
            id_test_dataset._data_list = None
        val_dataset._data_list = None
        test_dataset._data_list = None

        return {'train': train_dataset, 'id_val': id_val_dataset, 'id_test': id_test_dataset,
                'val': val_dataset, 'test': test_dataset, 'task': task,
                'metric': train_dataset.metric}, meta_info
