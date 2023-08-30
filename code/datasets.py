import json
import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd
import scipy
import torch
import yaml
from torch.utils import data as torchdata

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

PARSED_DATA_DIR = config['parsed_data_dir']


def dummy_encode(df, *col_names):
    """
    Dummy encode one or more categorical features in a pandas dataframe (uses first value as reference).
    Using dummy encoding rather than one-hot avoids co-linearity
    :param df: the pandas DataFrame
    :param col_names: the names of the column
    """

    for col_name in col_names:
        df = pd.concat([df, pd.get_dummies(df[col_name], prefix=col_name, drop_first=True)], axis=1)
        df.drop([col_name], axis=1, inplace=True)

    return df


def scipy_sparse_to_torch_sparse(matrix: scipy.sparse.spmatrix, device: torch.device = torch.device('cpu'),
                                 dtype=torch.float) -> torch.sparse_coo_tensor:
    """
    Convert a sparse scipy matrix to a torch sparse COO tensor
    Args:
        matrix: the scipy matrix
        device: the device to put the torch tensor on
        dtype: the type to convert the matrix to
    Returns: the torch sparse tensor on the correct device
    """
    coo_matrix = matrix.tocoo()
    values = coo_matrix.data
    indices = np.vstack((coo_matrix.row, coo_matrix.col))

    return torch.sparse_coo_tensor(torch.tensor(indices, dtype=torch.long), torch.tensor(values, dtype=dtype),
                                   torch.Size(coo_matrix.shape), device=device)


class Dataset(torchdata.Dataset):
    name = ''
    samples = 0
    max_choice_set_size = 0
    n_item_features = 0
    n_chooser_features = 0
    n_choosers = 0
    n_items = 0
    chooser_df = None
    item_df = None
    item_feature_names = None
    choice_df = None
    choices = None
    choice_sets = None
    choice_set_sizes = None
    choice_set_features = None
    chooser_features = None
    choosers = None
    categorical_chooser_features = None
    categorical_item_features = None
    chooser_graph = None
    item_id_feature = None
    choice_counts = None
    node_features = None

    def __init__(self, name, info_file='info.json'):
        """
        Load the dataset from standard format, located at {PARSED_DATA_DIR}/{name}.
        If you want to load a variant of the dataset, specify a different info_file
        :param name: the name of the dataset
        :param info_file: file with metadata about the dataset, located in {PARSED_DATA_DIR}/{name}/meta
        """
        # pickle_path = f'{PARSED_DATA_DIR}/pickles/{name}.pt'
        # if os.path.isfile(pickle_path):
        #     with open(pickle_path, 'rb') as f:
        #         data = torch.load(f)
        #
        #     self.__dict__.update(data.__dict__)
        #     return

        self.name = name
        self.device = torch.device('cpu')

        dir_path = f'{PARSED_DATA_DIR}/{name}'
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f'No directory {dir_path}. Make sure to configure the data path in config.yml.')

        with open(f'{dir_path}/meta/{info_file}', 'r') as f:
            self.info = json.load(f)

        self.samples = self.info['samples']
        self.max_choice_set_size = self.info['max_choice_set_size']
        self.categorical_chooser_features = self.info['categorical_chooser_features']
        self.categorical_item_features = self.info['categorical_item_features']
        self.item_id_feature = self.info['item_id_feature']

        self.n_chooser_features = int(self.info['chooser_features'])
        if self.n_chooser_features > 0:
            self.chooser_df = pd.read_csv(f'{dir_path}/choosers.txt', delimiter=' ', dtype=str).apply(pd.to_numeric, errors='ignore').set_index('chooser')
            self.chooser_df = dummy_encode(self.chooser_df, *self.categorical_chooser_features)
            self.n_chooser_features = len(self.chooser_df.columns)

        self.item_df = pd.read_csv(f'{dir_path}/items.txt', delimiter=' ').set_index('item')

        self.item_df = dummy_encode(self.item_df, *self.categorical_item_features)
        self.n_items = len(self.item_df)
        self.n_item_features = 0 if not self.info['item_features'] else len(self.item_df.columns)
        self.item_feature_names = [] if not self.info['item_features'] else list(self.item_df.columns)

        self.choice_df = pd.read_csv(f'{dir_path}/choices.txt', delimiter=' ', dtype=str).apply(pd.to_numeric, errors='ignore')

        jagged_choice_sets = [df.values for name, df in self.choice_df.groupby('sample')['item']]

        self.choice_set_sizes = torch.tensor([len(choice_set) for choice_set in jagged_choice_sets])
        self.choice_sets = torch.zeros((self.samples, self.max_choice_set_size), dtype=int)
        self.choices = torch.tensor([df.tolist().index(1) for name, df in self.choice_df.groupby('sample')['chosen']])

        item_id_feature_idx = None
        if self.item_id_feature is not None:
            item_id_feature_idx = self.item_feature_names.index(self.item_id_feature)

        if self.n_item_features > 0:
            self.choice_set_features = torch.zeros((self.samples, self.max_choice_set_size, self.n_item_features))

            # Convert item_id to a [0, n) index
            if self.item_id_feature is not None:
                self.reindex_item_id_map = {item_id: i for i, item_id in enumerate(self.item_df[self.item_id_feature].unique())}
                self.item_df[self.item_id_feature] = self.item_df[self.item_id_feature].map(self.reindex_item_id_map)

            item_features = torch.from_numpy(self.item_df.values)

        for i in range(self.samples):
            self.choice_sets[i, :self.choice_set_sizes[i]] = torch.from_numpy(jagged_choice_sets[i])
            if self.n_item_features > 0:
                self.choice_set_features[i, :self.choice_set_sizes[i]] = item_features[self.choice_sets[i, :self.choice_set_sizes[i]]]
                if self.item_id_feature is not None:
                    self.choice_sets[i, :self.choice_set_sizes[i]] = item_features[self.choice_sets[i, :self.choice_set_sizes[i]], item_id_feature_idx]

        if self.item_id_feature is not None:
            self.choice_set_features = self.choice_set_features[:, :, torch.arange(self.n_item_features) != item_id_feature_idx]
            self.n_item_features -= 1
            self.n_items = self.item_df[self.item_id_feature].nunique()
            self.item_feature_names.remove(self.item_id_feature)

        self.reindex_choosers_map = {chooser: i for i, chooser in enumerate(self.choice_df['chooser'].unique())}
        self.n_choosers = len(self.reindex_choosers_map)
        self.choosers = torch.from_numpy(self.choice_df[['sample', 'chooser']].drop_duplicates()['chooser'].map(self.reindex_choosers_map).values)

        if self.n_item_features > 0:
            self._normalize_item_features()

        if self.n_chooser_features > 0:
            self.chooser_df = self.chooser_df.rename(self.reindex_choosers_map)

            self.chooser_features = torch.from_numpy(self.chooser_df.loc[self.choosers].values).float()
            self.node_features = torch.from_numpy(self.chooser_df.loc[np.arange(self.n_choosers)].values).float()
            self._normalize_chooser_features()

        node_type = type(next(iter(self.reindex_choosers_map)))
        if os.path.isfile(f'{dir_path}/chooser-graph.txt'):
            self.chooser_graph = nx.read_edgelist(f'{dir_path}/chooser-graph.txt', nodetype=node_type)
            self.chooser_graph.add_nodes_from(self.choice_df['chooser'].unique())
            self.chooser_graph = nx.relabel_nodes(self.chooser_graph.subgraph(self.choice_df['chooser'].unique()), self.reindex_choosers_map)

        if os.path.isfile(f'{dir_path}/timestamped_edges.txt'):
            self.timestamped_edges = np.loadtxt(f'{dir_path}/timestamped_edges.txt', dtype=np.int64)
            self.timestamped_edges[:, 0] = [self.reindex_choosers_map[chooser] for chooser in self.timestamped_edges[:, 0]]
            self.timestamped_edges[:, 1] = [self.reindex_choosers_map[chooser] for chooser in self.timestamped_edges[:, 1]]
            self.chooser_graph = nx.Graph()
            self.chooser_graph.add_edges_from(self.timestamped_edges[:, :2])

        if os.path.isfile(f'{dir_path}/social-edges.txt'):
            self.social_graph = nx.read_edgelist(f'{dir_path}/social-edges.txt', nodetype=node_type)
            self.social_graph.add_nodes_from(self.choice_df['chooser'].unique())
            self.social_graph = nx.relabel_nodes(self.social_graph.subgraph(self.choice_df['chooser'].unique()), self.reindex_choosers_map)

        if os.path.isfile(f'{dir_path}/choice-counts.txt'):
            self.choice_counts = torch.from_numpy(np.loadtxt(f'{dir_path}/choice-counts.txt', dtype=np.int64, skiprows=1)[:, 1]).float()

        # with open(pickle_path, 'wb') as f:
        #     torch.save(self, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    def _normalize_item_features(self):
        choice_set_mask = torch.arange(self.max_choice_set_size)[None, :] < self.choice_set_sizes[:, None]
        all_feature_vecs = self.choice_set_features[choice_set_mask]
        means = all_feature_vecs.mean(0)
        stds = all_feature_vecs.std(0)
        self.choice_set_features[choice_set_mask] -= means
        self.choice_set_features[choice_set_mask] = torch.nan_to_num(self.choice_set_features[choice_set_mask] / stds)

    def _normalize_chooser_features(self):
        self.chooser_feature_means = self.chooser_features.mean(0)
        self.chooser_feature_stds = self.chooser_features.std(0)
        self.chooser_features -= self.chooser_feature_means
        self.chooser_features /= self.chooser_feature_stds

        self.node_feature_means = self.node_features.mean(0)
        self.node_feature_stds = self.node_features.std(0)
        self.node_features -= self.node_feature_means
        self.node_features /= self.node_feature_stds

    def shuffle(self):
        shuffle_idx = torch.randperm(self.samples)

        self.choices = self.choices[shuffle_idx]
        self.choice_sets = self.choice_sets[shuffle_idx]
        self.choice_set_sizes = self.choice_set_sizes[shuffle_idx]
        self.choosers = self.choosers[shuffle_idx]

        if self.n_item_features > 0:
            self.choice_set_features = self.choice_set_features[shuffle_idx]

        if self.choice_counts is not None:
            self.choice_counts = self.choice_counts[shuffle_idx]

    def to(self, device):
        self.device = device
        self.choices = self.choices.to(device)
        self.choice_sets = self.choice_sets.to(device)
        self.choice_set_sizes = self.choice_set_sizes.to(device)
        self.choosers = self.choosers.to(device)

        if self.n_item_features > 0:
            self.choice_set_features = self.choice_set_features.to(device)

        if self.n_chooser_features > 0:
            self.chooser_features = self.chooser_features.to(device)

        if self.choice_counts is not None:
            self.choice_counts = self.choice_counts.to(device)

        if self.node_features is not None:
            self.node_features = self.node_features.to(device)

        return self

    def __getitem__(self, i):
        if self.n_item_features == 0 and self.n_chooser_features > 0 and self.choice_counts is not None:
            return self.choice_sets[i], self.choice_set_sizes[i], self.choosers[i], self.choices[i], self.chooser_features[i], self.choice_counts[i]

        if self.n_item_features > 0:
            return self.choice_sets[i], self.choice_set_sizes[i], self.choosers[i], self.choice_set_features[i], self.choices[i]
        else:
            return self.choice_sets[i], self.choice_set_sizes[i], self.choosers[i], self.choices[i]

    def __len__(self):
        return len(self.choices)

    def get_normalized_A(self, graph) -> torch.Tensor:
        """
        Compute the (sparse) normalized adjacency matrix of thr graph for use in a GCN
        Returns: the normalized adj matrix
        """
        node_order = np.arange(self.n_choosers)
        tilde_A = nx.adjacency_matrix(graph, nodelist=node_order) + scipy.sparse.eye(self.n_choosers)
        degree_view = graph.degree()
        degrees = np.array([degree_view[node] + 2 for node in node_order])  # need to add 2 to degree for self loops
        tilde_D_minus_1_2 = scipy.sparse.diags(degrees).power(-0.5)
        return scipy_sparse_to_torch_sparse(tilde_D_minus_1_2 @ tilde_A @ tilde_D_minus_1_2, device=self.device)


if __name__ == '__main__':
    for name in ['app-install', 'app-usage', 'election-2016', 'ca-election-2016', 'ca-election-2020']:
        data = Dataset(name)
        print(name)
        if data.choice_counts is None:
            print('\tSamples:', data.samples)
        else:
            print('\tSamples:', data.choice_counts.sum().item())
        print('\tChoosers:', data.n_choosers)
        print('\tItems:', data.n_items)
        print('\tChooser Features:', data.n_chooser_features)
        print('\tItem Features:', data.n_item_features, data.item_feature_names)
        print('\tChoice Set Size:', data.choice_set_sizes.min().item(), '-', data.choice_set_sizes.max().item())
        print()
