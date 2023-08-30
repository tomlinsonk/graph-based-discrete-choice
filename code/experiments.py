import argparse
import time
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from torch.multiprocessing import Pool
from torch.utils.data import Subset
from tqdm import tqdm

import choice_models
from datasets import Dataset

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

RESULTS_DIR = config['results_dir']
RAW_DATA_DIR = config['data_dir']
PARSED_DATA_DIR = config['parsed_data_dir']

CPU = torch.device('cpu')


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', dest='gpu', type=int, default=-1, help='which GPU to run on (default: CPU)')
parser.add_argument('--dataset-batch', dest='dataset_batch', type=int, default=0, help='which dataset batch to run on (default: 0)')
parser.add_argument('--masked', dest='masked', action='store_true', default=False, help='whether to run the masked node experiment')
parser.add_argument('--per-item-utilities', dest='per_item_utilities', action='store_true', default=False, help='whether to learn per-item utilities')
parser.add_argument('--timestamped-edges', dest='timestamped_edges', action='store_true', default=False, help='whether the datasets have timestamped edges')
parser.add_argument('--election-adjacency', dest='election_adjacency', default=False, help='run the election experiment with the adjacency graph')
parser.add_argument('--election-social', dest='election_social', default=False, help='run the election experiment with the social graph')
parser.add_argument('--propagation', dest='propagation', action='store_true', default=False, help='run parameter propagation for electon experiment')
parser.add_argument('--network-convergence', dest='network_convergence', action='store_true', default=False, help='run simulated network convergence experiment')
parser.add_argument('--threads', dest='threads', type=int, default=False, help='number of threads')
parser.add_argument('--dataset', dest='dataset', type=str, nargs='+', help='datasets to run on')
parser.add_argument('--app-prox-thresholds', dest='app_prox_thresholds', type=int, nargs='+', help='what thresholds to use for building app proximity graph')
parser.add_argument('--timing', action='store_true', help='run the timing experiment')


args = parser.parse_args()


def election_experiment_helper(args):

    seed, train_frac, l2_lambda, lr, train_idx, val_idx, test_idx, laplace_lambdas, dataset, is_adjacency = args

    data = Dataset(dataset).to(choice_models.device)

    train_set = Subset(data, train_idx)
    val_set = Subset(data, val_idx)
    test_set = Subset(data, test_idx)

    if is_adjacency:
        graph = data.chooser_graph
    else:
        graph = data.social_graph

    edges = torch.tensor(list(graph.edges()))

    mnl_init_args = (data.n_chooser_features, data.n_items)
    pc_mnl_init_args = (data.n_chooser_features, data.n_items, data.n_choosers)
    gcn_init_args = (data.node_features, data.n_items, data.get_normalized_A(graph), 16, 16, 0.5)

    sample_count_idx = 5
    choice_idx = 3
    mnl_arg_idxs = (0, 1, 4)
    pc_mnl_arg_idxs = (0, 1, 4, 2)
    gcn_arg_idxs = (0, 1, 2)

    # print(f'train frac: {train_frac}, l2_lambda: {l2_lambda}, lr: {lr}')

    gcn_results = dict()
    mnl_results = dict()
    pc_mnl_results = dict()
    laplace_results = dict()

    for Model, init_args, results, arg_idxs in (
            (choice_models.GCNMultinomialLogit, gcn_init_args, gcn_results, gcn_arg_idxs),
            (choice_models.MultinomialLogit, mnl_init_args, mnl_results, mnl_arg_idxs),
            (choice_models.PerChooserMultinomialLogit, pc_mnl_init_args, pc_mnl_results, pc_mnl_arg_idxs)
    ):
        # print('\t', Model)
        model = Model(*init_args).to(choice_models.device)

        train_losses, val_losses, gradients = choice_models.train(model, train_set,
                                                                  val_set,
                                                                  arg_idxs,
                                                                  choice_idx,
                                                                  show_live_loss=False,
                                                                  show_progress=False,
                                                                  l2_lambda=l2_lambda,
                                                                  learning_rate=lr,
                                                                  sample_count_idx=sample_count_idx)

        test_loss, test_acc, test_mrr = choice_models.test(model, test_set, arg_idxs, choice_idx,
                                                 sample_count_idx=sample_count_idx)

        results['best_model', train_frac, seed] = test_loss, test_acc, test_mrr, model.cpu().state_dict(), l2_lambda, lr, val_losses[-1]
        results[train_frac, l2_lambda, lr, seed] = train_losses, val_losses, gradients, test_loss, test_acc, test_mrr

    laplace_results['best_model', train_frac, seed] = None, None, None, None, None, np.inf
    for laplace_lambda in laplace_lambdas:
        # print(f'\tlaplace {laplace_lambda}')

        laplace_mnl = choice_models.PerChooserMultinomialLogit(*pc_mnl_init_args).to(choice_models.device)
        laplace_train_losses, laplace_val_losses, laplace_gradients = choice_models.train(laplace_mnl,
                                                                                          train_set,
                                                                                          val_set,
                                                                                          pc_mnl_arg_idxs,
                                                                                          choice_idx,
                                                                                          edges=edges,
                                                                                          laplace_lambda=laplace_lambda,
                                                                                          l2_lambda=l2_lambda,
                                                                                          learning_rate=lr,
                                                                                          show_live_loss=False,
                                                                                          show_progress=False,
                                                                                          sample_count_idx=sample_count_idx)

        laplace_test_loss, laplace_test_acc, laplace_test_mrr = choice_models.test(laplace_mnl, test_set, pc_mnl_arg_idxs, choice_idx,
                                                                 sample_count_idx=sample_count_idx)

        if laplace_val_losses[-1] < laplace_results['best_model', train_frac, seed][-1]:
            laplace_results[
                'best_model', train_frac, seed] = laplace_test_loss, laplace_test_acc, laplace_test_mrr, laplace_mnl.cpu().state_dict(), l2_lambda, lr, laplace_lambda, \
                                                  laplace_val_losses[-1]

        laplace_results[train_frac, laplace_lambda, l2_lambda, lr, seed] = laplace_train_losses, laplace_val_losses, laplace_gradients, laplace_test_loss, laplace_test_acc, laplace_test_mrr

    return train_frac, seed, l2_lambda, lr, gcn_results, mnl_results, pc_mnl_results, laplace_results


def election_experiment(args):
    print('Running election experiment...')
    dataset = args.election_social if args.election_social else args.election_adjacency
    data = Dataset(dataset).to(choice_models.device)

    l2_lambdas = np.logspace(-1, -5, 5)
    # lrs = [0.9, 0.5, 0.1]
    lrs = [0.001, 0.01, 0.1]
    laplace_lambdas = np.logspace(-2, -5, 4)
    train_fracs = np.linspace(0.1, 0.8, 8)
    seeds = range(8)

    mnl_results = dict()
    pc_mnl_results = dict()
    gcn_results = dict()

    laplace_results = dict()
    all_counties = dict()
    all_idxs = dict()

    propagation_results = dict()

    counties = torch.unique(data.choosers)

    def get_params():

        for seed in seeds:
            torch.random.manual_seed(seed)

            for train_frac in train_fracs:
                train_counties, val_test_counties = train_test_split(counties, train_size=train_frac)
                val_counties, test_counties = train_test_split(val_test_counties, train_size=0.5)

                train_idx = (data.choosers[..., None] == train_counties).any(-1).nonzero().squeeze()
                val_idx = (data.choosers[..., None] == val_counties).any(-1).nonzero().squeeze()
                test_idx = (data.choosers[..., None] == test_counties).any(-1).nonzero().squeeze()

                all_counties[train_frac, seed] = train_counties, val_counties, test_counties
                all_idxs[train_frac, seed] = train_idx, val_idx, test_idx

                mnl_results['best_model', train_frac, seed] = None, None, None, None, np.inf
                pc_mnl_results['best_model', train_frac, seed] = None, None, None, None, np.inf
                gcn_results['best_model', train_frac, seed] = None, None, None, None, None, np.inf
                laplace_results['best_model', train_frac, seed] = None, None, None, None, None, np.inf

                if args.election_adjacency:
                    graph = data.chooser_graph
                else:
                    graph = data.social_graph

                print(f'Running propagation (seed {seed}, train_frac {train_frac})')
                propagation_results[train_frac, seed] = choice_models.choice_propagation(Subset(data, train_idx), Subset(data, test_idx), graph, 3, data.n_choosers, data.n_items, 5)

                for l2_lambda in l2_lambdas:
                    for lr in lrs:
                        yield seed, train_frac, l2_lambda, lr, train_idx, val_idx, test_idx, laplace_lambdas, dataset, args.election_adjacency


    with Pool(args.threads) as pool:
        for train_frac, seed, l2_lambda, lr, gcn_trial_results, mnl_trial_results, pc_mnl_trial_results, laplace_trial_results in \
                tqdm(pool.imap_unordered(election_experiment_helper, get_params()),
                     total=len(seeds)*len(train_fracs)*len(l2_lambdas)*len(lrs)):

            for trial_results, results in zip((gcn_trial_results, mnl_trial_results, pc_mnl_trial_results, laplace_trial_results),
                                              (gcn_results, mnl_results, pc_mnl_results, laplace_results)):

                if trial_results['best_model', train_frac, seed][-1] < results['best_model', train_frac, seed][-1]:
                    results['best_model', train_frac, seed] = trial_results['best_model', train_frac, seed]

                results.update({key: val for key, val in trial_results.items() if key[0] != 'best_model'})

    savefile_name = f'{RESULTS_DIR}/{data.name}-masked-laplace-results-{"adjacency" if args.election_adjacency else "social"}.pt'
    with open(savefile_name, 'wb') as f:
        torch.save((mnl_results, pc_mnl_results, gcn_results, laplace_results, propagation_results, train_fracs, laplace_lambdas, l2_lambdas, lrs, all_counties, all_idxs, seeds), f)
    print('Saved results to:', savefile_name)
    del data


def election_propagation_helper(args):
    cl_args, train_frac, seed, train_idx, val_idx, test_idx = args

    choice_idx = 3
    count_idx = 5

    dataset = cl_args.election_social if cl_args.election_social else cl_args.election_adjacency
    data = Dataset(dataset)
    graph = data.chooser_graph if cl_args.election_adjacency else data.social_graph

    return train_frac, seed, choice_models.choice_propagation(Subset(data, train_idx), Subset(data, test_idx), graph,
                                                              choice_idx, data.n_choosers, data.n_items, count_idx,
                                                              val_set=Subset(data, val_idx))


def election_parallel_propagation(args):
    dataset = args.election_social if args.election_social else args.election_adjacency

    choice_results_name = f'{RESULTS_DIR}/{dataset}-masked-laplace-results-{"adjacency" if args.election_adjacency else "social"}.pt'
    with open(choice_results_name, 'rb') as f:
        _, _, _, _, _, train_fracs, _, _, _, all_counties, all_idxs, seeds = torch.load(f)

    # param: (args, train_idx, val_idx, test_idx)
    params = [(args, train_frac, seed, *all_idxs[train_frac, seed]) for train_frac in train_fracs for seed in seeds]

    all_results = dict()
    with Pool(args.threads) as pool:
        for train_frac, seed, propagation_results in tqdm(pool.imap_unordered(election_propagation_helper, params),
                                                          total=len(params)):
            all_results[train_frac, seed] = propagation_results

    with open(f'{RESULTS_DIR}/{dataset}-prop-results-{"adjacency" if args.election_adjacency else "social"}.pt', 'wb') as f:
        torch.save(all_results, f)


def app_propagation_helper(args):
    cl_args, dataset, train_frac, seed, train_idx, val_idx, test_idx = args
    choice_idx = 4 if dataset == 'app-usage' else 3
    data = Dataset(dataset)

    return train_frac, seed, choice_models.choice_propagation(Subset(data, train_idx), Subset(data, test_idx), data.chooser_graph,
                                                              choice_idx, data.n_choosers, data.n_items,
                                                              val_set=Subset(data, val_idx))


def app_parallel_propagation(dataset, args):
    for thresh in args.app_prox_thresholds:
        write_app_graph_file(dataset, thresh)

        choice_results_name = f'{RESULTS_DIR}/{dataset}-masked-laplace-results-per-item-prox-threshold-{thresh}.pt'
        with open(choice_results_name, 'rb') as f:
            _, _, _, _, _, train_fracs, _, _, _, _, all_idxs, seeds = torch.load(f)

        # param: (args, train_idx, val_idx, test_idx)
        params = [(args, dataset, train_frac, seed, *all_idxs[train_frac, seed]) for train_frac in train_fracs for seed in seeds]

        all_results = dict()
        with Pool(args.threads) as pool:
            for train_frac, seed, propagation_results in tqdm(pool.imap_unordered(app_propagation_helper, params),
                                                              total=len(params)):
                all_results[train_frac, seed] = propagation_results

        with open(f'{RESULTS_DIR}/{dataset}-prop-results-per-item-prox-threshold-{thresh}.pt', 'wb') as f:
            torch.save(all_results, f)


def app_experiment_helper(args):
    seed, train_frac, l2_lambda, lr, train_idx, val_idx, test_idx, laplace_lambdas, dataset, timestamped_edges, per_item_utilities = args

    data = Dataset(dataset).to(choice_models.device)
    edges = torch.from_numpy(data.timestamped_edges[:, :2]) if timestamped_edges else torch.tensor(list(data.chooser_graph.edges()))

    train_set = Subset(data, train_idx)
    val_set = Subset(data, val_idx)
    test_set = Subset(data, test_idx)

    normalized_A = data.get_normalized_A(data.chooser_graph)

    dropout_p = 0.5
    embedding_dim = 16
    hidden_dim = 16
    output_dim = 16

    if data.n_item_features > 0:
        global_init_args = (data.n_item_features,) + ((data.n_items,) if per_item_utilities else ())
        pc_init_args = (data.n_item_features, data.n_choosers) + ((data.n_items,) if per_item_utilities else ())
        gcn_init_args = (data.n_items, data.n_item_features, data.n_choosers, normalized_A, embedding_dim, hidden_dim, output_dim, dropout_p)

        # self.choice_sets[i], self.choice_set_sizes[i], self.choosers[i], self.choice_set_features[i], self.choices[i]
        choice_idx = 4
        global_arg_idxs = (3, 1) + ((0,) if per_item_utilities else ())
        pc_arg_idxs = (3, 1, 2) + ((0,) if per_item_utilities else ())
        gcn_arg_idxs = (3, 0, 1, 2)

        global_model = choice_models.ConditionalLogit
        pc_model = choice_models.PerChooserConditionalLogit
        gcn_model = choice_models.GCNEmbeddingConditionalLogit
    else:

        # self.choice_sets[i], self.choice_set_sizes[i], self.choosers[i], self.choices[i]
        global_init_args = (data.n_items,)
        pc_init_args = (data.n_items, data.n_choosers)
        gcn_init_args = (data.n_choosers, data.n_items, normalized_A, embedding_dim, hidden_dim, output_dim, dropout_p)

        choice_idx = 3
        global_arg_idxs = (0, 1)
        pc_arg_idxs = (0, 1, 2)
        gcn_arg_idxs = (0, 1, 2)

        global_model = choice_models.Logit
        pc_model = choice_models.PerChooserLogit
        gcn_model = choice_models.GCNEmbeddingLogit

    global_results = dict()
    pc_results = dict()
    gcn_results = dict()
    laplace_results = dict()

    global_results['best_model', train_frac, seed] = None, None, None, None, np.inf
    pc_results['best_model', train_frac, seed] = None, None, None, None, np.inf
    gcn_results['best_model', train_frac, seed] = None, None, None, None, np.inf
    laplace_results['best_model', train_frac, seed] = None, None, None, None, None, np.inf

    for Model, init_args, results, arg_idxs in (
            (gcn_model, gcn_init_args, gcn_results, gcn_arg_idxs),
            (global_model, global_init_args, global_results, global_arg_idxs),
            (pc_model, pc_init_args, pc_results, pc_arg_idxs)):
        # print('\t', Model)
        model = Model(*init_args).to(choice_models.device)

        train_losses, val_losses, gradients = choice_models.train(model, train_set,
                                                                  val_set,
                                                                  arg_idxs,
                                                                  choice_idx,
                                                                  show_live_loss=False,
                                                                  show_progress=False,
                                                                  l2_lambda=l2_lambda,
                                                                  learning_rate=lr,
                                                                  val_increase_break=False)

        test_loss, test_acc, test_mrr = choice_models.test(model, test_set, arg_idxs, choice_idx)

        if val_losses[-1] < results['best_model', train_frac, seed][-1]:
            results[
                'best_model', train_frac, seed] = test_loss, test_acc, test_mrr, model.cpu().state_dict(), l2_lambda, lr, \
                                                  val_losses[-1]

        results[train_frac, l2_lambda, lr, seed] = train_losses, val_losses, gradients, test_loss, test_acc, test_mrr

    for laplace_lambda in laplace_lambdas:
        # print(f'\tlaplace {laplace_lambda}')

        laplace_mnl = pc_model(*pc_init_args).to(
            choice_models.device)
        laplace_train_losses, laplace_val_losses, laplace_gradients = choice_models.train(laplace_mnl,
                                                                                          train_set,
                                                                                          val_set,
                                                                                          pc_arg_idxs,
                                                                                          choice_idx,
                                                                                          edges=edges,
                                                                                          laplace_lambda=laplace_lambda,
                                                                                          l2_lambda=l2_lambda,
                                                                                          learning_rate=lr,
                                                                                          show_live_loss=False,
                                                                                          show_progress=False,
                                                                                          val_increase_break=False)

        laplace_test_loss, laplace_test_acc, laplace_test_mrr = choice_models.test(laplace_mnl, test_set, pc_arg_idxs, choice_idx)

        if laplace_val_losses[-1] < laplace_results['best_model', train_frac, seed][-1]:
            laplace_results[
                'best_model', train_frac, seed] = laplace_test_loss, laplace_test_acc, laplace_test_mrr, laplace_mnl.cpu().state_dict(), l2_lambda, lr, laplace_lambda, laplace_val_losses[-1]

        laplace_results[
            train_frac, laplace_lambda, l2_lambda, lr, seed] = laplace_train_losses, laplace_val_losses, laplace_gradients, laplace_test_loss, laplace_test_acc, laplace_test_mrr

    return train_frac, seed, l2_lambda, lr, gcn_results, global_results, pc_results, laplace_results


def app_experiment(dataset_name, args, app_prox_thresholds=None):
    print(f'Running masked_nodes_experiment({dataset_name}, per_item_utilities={args.per_item_utilities}, timestamped_edges={args.timestamped_edges}), app_prox_thresholds={app_prox_thresholds}')

    data = Dataset(dataset).to(choice_models.device)

    laplace_lambdas = np.logspace(0, -7, 8)
    l2_lambdas = np.concatenate((np.logspace(-1, -5, 5), [0]))
    lrs = [0.001, 0.01, 0.1]
    train_fracs = np.linspace(0.1, 0.8, 8)
    seeds = range(64)

    global_results = dict()
    pc_results = dict()
    gcn_results = dict()
    laplace_results = dict()
    propagation_results = dict()

    chooser_splits = dict()
    all_idxs = dict()

    unique_choosers = torch.unique(data.choosers)

    def get_params():
        for seed in seeds:
            torch.random.manual_seed(seed)

            for train_frac in train_fracs:
                train_choosers, val_test_choosers = train_test_split(unique_choosers, train_size=train_frac)
                val_choosers, test_choosers = train_test_split(val_test_choosers, train_size=0.5)

                train_idx = (data.choosers[..., None] == train_choosers).any(-1).nonzero().squeeze()
                val_idx = (data.choosers[..., None] == val_choosers).any(-1).nonzero().squeeze()
                test_idx = (data.choosers[..., None] == test_choosers).any(-1).nonzero().squeeze()

                chooser_splits[train_frac, seed] = train_choosers, val_choosers, test_choosers
                all_idxs[train_frac, seed] = train_idx, val_idx, test_idx

                global_results['best_model', train_frac, seed] = None, None, None, None, np.inf
                pc_results['best_model', train_frac, seed] = None, None, None, None, np.inf
                gcn_results['best_model', train_frac, seed] = None, None, None, None, np.inf
                laplace_results['best_model', train_frac, seed] = None, None, None, None, None, np.inf

                propagation_results[train_frac, seed] = choice_models.choice_propagation(Subset(data, train_idx), Subset(data, test_idx), data.chooser_graph, -1, data.n_choosers, data.n_items)

                for l2_lambda in l2_lambdas:

                    for lr in lrs:
                        yield seed, train_frac, l2_lambda, lr, train_idx, val_idx, test_idx, laplace_lambdas, \
                              dataset_name, args.timestamped_edges, args.per_item_utilities

    def run(savefile):
        with Pool(args.threads) as pool:
            for train_frac, seed, l2_lambda, lr, gcn_trial_results, global_trial_results, pc_trial_results, laplace_trial_results in \
                    tqdm(pool.imap_unordered(app_experiment_helper, get_params()),
                         total=len(seeds) * len(train_fracs) * len(l2_lambdas) * len(lrs)):

                for trial_results, results in zip(
                        (gcn_trial_results, global_trial_results, pc_trial_results, laplace_trial_results),
                        (gcn_results, global_results, pc_results, laplace_results)):

                    if trial_results['best_model', train_frac, seed][-1] < results['best_model', train_frac, seed][-1]:
                        results['best_model', train_frac, seed] = trial_results['best_model', train_frac, seed]

                    results.update({key: val for key, val in trial_results.items() if key[0] != 'best_model'})

        with open(savefile, 'wb') as f:
            torch.save((global_results, pc_results, gcn_results, laplace_results, propagation_results, train_fracs, laplace_lambdas, l2_lambdas,
                        lrs, chooser_splits, all_idxs, seeds), f)

    if app_prox_thresholds is None:
        savefile = f'{RESULTS_DIR}/{data.name}-masked-laplace-results{"-per-item" if args.per_item_utilities else ""}.pt'
        run(savefile)
    else:
        for thresh in app_prox_thresholds:
            write_app_graph_file(dataset, thresh)
            savefile = f'{RESULTS_DIR}/{data.name}-masked-laplace-results{"-per-item" if args.per_item_utilities else ""}-prox-threshold-{thresh}.pt'
            print('running with threshold', thresh)
            run(savefile)


def write_app_graph_file(dataset, thresh):
    print('reading csv')
    proximity_df = pd.read_csv(f'{RAW_DATA_DIR}/friends-and-family/BluetoothProximity.csv').dropna()
    proximity_df['date'] = pd.to_datetime(proximity_df['date'])

    pair_cols = ['participantID', 'participantID.B']

    print('looping')

    edges = ''
    for participant in tqdm(proximity_df['participantID'].unique()):
        pairs = proximity_df[proximity_df['participantID'] == participant][pair_cols].apply(tuple,
                                                                                            axis='columns').values
        counts = Counter(pairs)
        edges += '\n'.join(f'{participant} {pair[1]}' for pair, count in counts.most_common(thresh)) + '\n'

    print('writing')

    with open(f'{PARSED_DATA_DIR}/{dataset}/chooser-graph.txt', 'w') as f:
        f.write(edges)


# https://stackoverflow.com/a/55317373/6866505
def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)


def _network_convergence_helper(args):
    trial, laplace_reg, items, n, k, p, sample_range, chooser_range, lambda_range = args
    np.random.seed(trial)
    torch.random.manual_seed(trial)

    graph = nx.connected_watts_strogatz_graph(n, k, p)
    L = nx.laplacian_matrix(graph, chooser_range).todense()
    I = np.eye(items)
    edges = torch.tensor(list(graph.edges)).to(choice_models.device) if laplace_reg else None

    results = dict()

    for l in lambda_range[::-1]:
        cov = np.linalg.inv(l * np.kron(I, L) + np.eye(items * n))

        utilities = np.random.multivariate_normal(np.zeros(items * n), cov).reshape(n, items, order='F')
        laplace_lambda = l / 2 if laplace_reg else None

        # For identifiability, fix utility of item 0 to 0
        utilities -= utilities[:, 0][:, None]

        mses = []

        for samples_per_chooser in sample_range:
            samples = n * samples_per_chooser

            choosers = np.tile(np.arange(n), samples_per_chooser)
            choice_set_sizes = np.random.randint(2, items + 1, samples)
            items_present = np.zeros((samples, items))
            items_present[np.arange(items)[None, :] < choice_set_sizes[:, None]] = 1

            items_present = shuffle_along_axis(items_present, 1)
            choice_sets = np.zeros((samples, items), dtype=int)

            rows, cols = items_present.nonzero()
            sum = 0

            observed_item_counts = np.zeros((n, items), dtype=int)

            for i in range(samples):
                choice_sets[i, :choice_set_sizes[i]] = cols[sum:sum + choice_set_sizes[i]]
                sum += choice_set_sizes[i]
                observed_item_counts[choosers[i], choice_sets[i, :choice_set_sizes[i]]] += 1

            item_utilities = utilities[choosers[:, None], choice_sets]
            item_utilities[np.arange(items)[None, :] >= choice_set_sizes[:, None]] = -np.inf

            prob = softmax(item_utilities, axis=1)

            # https://stackoverflow.com/a/47722393/6866505
            choices = (prob.cumsum(1) > np.random.rand(samples)[:, None]).argmax(1)

            choosers = torch.from_numpy(choosers).to(choice_models.device)
            choice_sets = torch.from_numpy(choice_sets).to(choice_models.device)
            choice_set_sizes = torch.from_numpy(choice_set_sizes).to(choice_models.device)
            choices = torch.from_numpy(choices).to(choice_models.device)

            data = [choosers, choice_sets, choice_set_sizes, choices]

            model = choice_models.PerChooserLogit(items, n).to(choice_models.device)
            choice_models.train(model, data, data, (1, 2, 0), 3, l2_lambda=0, learning_rate=0.001,
                                laplace_lambda=laplace_lambda, edges=edges, show_progress=True,
                                dont_break=True)

            pred_utils = model.utilities.detach().cpu().numpy()
            # For identifiability, fix utility of item 0 to 0
            pred_utils -= pred_utils[:, 0][:, None]

            # only measure MSE on observed items
            observed_item_mask = observed_item_counts > 0

            mse = np.square(utilities[observed_item_mask] - pred_utils[observed_item_mask]).mean()

            # print('MSE on all', np.square(utilities - pred_utils).mean(), 'on observed', mse)
            print(trial, l, samples_per_chooser, mse)

            mses.append(mse)

        results[trial, l] = mses

    return results


def network_correlation_convergence_experiment(laplace_reg=False):
    items = 20
    n = 100
    k = 5
    p = 0.1

    sample_range = np.logspace(0, 3, 10, dtype=int)
    chooser_range = range(n)
    lambda_range = [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01]

    trial_range = range(8)

    params = [(trial, laplace_reg, items, n, k, p, sample_range, chooser_range, lambda_range) for trial in trial_range]

    results = dict()

    with Pool(args.threads) as pool:
        for trial_results in tqdm(pool.map(_network_convergence_helper, params), total=len(params)):
            results.update(trial_results)

    with open(f'{RESULTS_DIR}/network-convergence-{"no-" if not laplace_reg else ""}laplace-reg.pt', 'wb') as f:
        torch.save((results, items, n, k, p, trial_range, sample_range, chooser_range, lambda_range), f)


def runtime_expriment():
    datasets = ['election-2016', 'ca-election-2016', 'app-usage', 'app-install', 'ca-election-2020']
    n_trials = 4
    train_frac = 0.5
    lr = 0.01
    l2_lambda = 0.001
    laplace_lambda = 0.0001
    alpha = 0.5
    np.random.seed(0)

    dropout_p = 0.5
    embedding_dim = 16
    hidden_dim = 16
    output_dim = 16

    models = {
        'app-usage': (choice_models.ConditionalLogit, choice_models.PerChooserConditionalLogit, choice_models.GCNEmbeddingConditionalLogit, 'Propagation'),
        'app-install': (choice_models.Logit, choice_models.PerChooserLogit, choice_models.GCNEmbeddingLogit, 'Propagation'),
        'election-2016': (choice_models.MultinomialLogit, choice_models.PerChooserMultinomialLogit, choice_models.GCNMultinomialLogit, 'Propagation'),
        'ca-election-2016': (choice_models.MultinomialLogit, choice_models.PerChooserMultinomialLogit, choice_models.GCNMultinomialLogit, 'Propagation'),
        'ca-election-2020': (choice_models.MultinomialLogit, choice_models.PerChooserMultinomialLogit, choice_models.GCNMultinomialLogit, 'Propagation'),
    }

    model_arg_idxs = {
        choice_models.Logit: (0, 1),
        choice_models.PerChooserLogit: (0, 1, 2),
        choice_models.GCNEmbeddingLogit: (0, 1, 2),
        choice_models.ConditionalLogit: (3, 1, 0),
        choice_models.PerChooserConditionalLogit: (3, 1, 2, 0),
        choice_models.GCNEmbeddingConditionalLogit: (3, 0, 1, 2),
        choice_models.MultinomialLogit: (0, 1, 4),
        choice_models.PerChooserMultinomialLogit: (0, 1, 4, 2),
        choice_models.GCNMultinomialLogit: (0, 1, 2)
    }

    laplace_models = {choice_models.PerChooserLogit, choice_models.PerChooserConditionalLogit, choice_models.PerChooserMultinomialLogit}
    gcn_models = {choice_models.GCNEmbeddingLogit, choice_models.GCNEmbeddingConditionalLogit, choice_models.GCNMultinomialLogit}

    times = {dataset: {model: [] for model in models[dataset]} for dataset in datasets}

    for dataset in datasets:
        data = Dataset(dataset)
        graph = data.social_graph if dataset == 'election-2016' else data.chooser_graph

        choice_idx = 4 if dataset == 'app-usage' else 3
        sample_count_idx = 5 if 'election' in dataset else None

        edges = torch.tensor(list(graph.edges()))
        unique_choosers = torch.unique(data.choosers)

        # Include time for computing normalized A in GCN time
        start = time.time()
        normalized_A = data.get_normalized_A(graph)
        normalized_A_time = time.time() - start

        train_choosers, val_test_choosers = train_test_split(unique_choosers, train_size=train_frac, random_state=0)
        val_choosers, test_choosers = train_test_split(val_test_choosers, train_size=0.5, random_state=0)

        train_idx = (data.choosers[..., None] == train_choosers).any(-1).nonzero().squeeze()
        val_idx = (data.choosers[..., None] == val_choosers).any(-1).nonzero().squeeze()
        test_idx = (data.choosers[..., None] == test_choosers).any(-1).nonzero().squeeze()

        train_set = Subset(data, train_idx)
        val_set = Subset(data, val_idx)
        test_set = Subset(data, test_idx)

        model_init_args = {
            choice_models.Logit: (data.n_items,),
            choice_models.PerChooserLogit: (data.n_items, data.n_choosers),
            choice_models.GCNEmbeddingLogit: (data.n_choosers, data.n_items, normalized_A, embedding_dim, hidden_dim, output_dim, dropout_p),
            choice_models.ConditionalLogit: (data.n_item_features, data.n_items),
            choice_models.PerChooserConditionalLogit:  (data.n_item_features, data.n_choosers, data.n_items),
            choice_models.GCNEmbeddingConditionalLogit: (data.n_items, data.n_item_features, data.n_choosers, normalized_A, embedding_dim, hidden_dim, output_dim, dropout_p),
            choice_models.MultinomialLogit: (data.n_chooser_features, data.n_items),
            choice_models.PerChooserMultinomialLogit: (data.n_chooser_features, data.n_items, data.n_choosers),
            choice_models.GCNMultinomialLogit: (data.node_features, data.n_items, normalized_A, hidden_dim, output_dim, dropout_p)
        }

        for trial in range(n_trials):
            for model_type in np.random.permutation(models[dataset]):
                print(dataset, trial, model_type)
                start = time.time()

                if model_type == 'Propagation':
                    choice_models.choice_propagation(train_set, test_set, graph, choice_idx, data.n_choosers, data.n_items, val_set=val_set, sample_count_idx=sample_count_idx, alphas=[alpha])
                else:
                    model = model_type(*model_init_args[model_type])
                    train_losses, val_losses, gradients = choice_models.train(
                        model, train_set,
                        val_set,
                        model_arg_idxs[model_type],
                        choice_idx,
                        show_live_loss=False,
                        show_progress=True,
                        l2_lambda=l2_lambda,
                        learning_rate=lr,
                        val_increase_break=False,
                        laplace_lambda=laplace_lambda,
                        edges=edges if model_type in laplace_models else None,
                        sample_count_idx=sample_count_idx)

                    test_loss, test_acc, test_mrr = choice_models.test(
                        model, test_set, model_arg_idxs[model_type], choice_idx, sample_count_idx)

                runtime = time.time() - start
                if model_type in gcn_models:
                    runtime += normalized_A_time

                print(runtime, 'seconds')

                times[dataset][model_type].append(runtime)

    with open(f'results/timing-results.pt', 'wb') as f:
        torch.save((datasets, models, times), f)


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    torch.multiprocessing.set_sharing_strategy('file_system')

    if args.gpu >= 0:
        choice_models.device = torch.device(f'cuda:{args.gpu}')
    else:
        choice_models.device = torch.device('cpu')

    if args.propagation:
        if args.election_social or args.election_adjacency:
            election_parallel_propagation(args)
        else:
            for dataset in args.dataset:
                app_parallel_propagation(dataset, args)

    elif args.election_social or args.election_adjacency:
        election_experiment(args)
    elif args.network_convergence:
        network_correlation_convergence_experiment(laplace_reg=True)
        network_correlation_convergence_experiment(laplace_reg=False)

    elif args.masked:
        for dataset in args.dataset:
            app_experiment(dataset, args, args.app_prox_thresholds)

    elif args.timing:
        runtime_expriment()


