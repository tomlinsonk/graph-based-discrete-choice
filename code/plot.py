import os
from collections import defaultdict

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import yaml
from cartopy.io.shapereader import Reader, natural_earth

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

import choice_models
from choice_models import PerChooserMultinomialLogit
from datasets import Dataset
from experiments import write_app_graph_file

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

RESULTS_DIR = config['results_dir']
DATA_DIR = config['data_dir']

state_fips_map = {
    '01': 'Alabama', '02': 'Alaska', '04': 'Arizona', '05': 'Arkansas', '06': 'California', '08': 'Colorado',
    '09': 'Connecticut', '10': 'Delaware', '12': 'Florida', '13': 'Georgia', '15': 'Hawaii', '16': 'Idaho',
    '17': 'Illinois', '18': 'Indiana', '19': 'Iowa', '20': 'Kansas', '21': 'Kentucky', '22': 'Louisiana',
    '23': 'Maine', '24': 'Maryland', '25': 'Massachusetts', '26': 'Michigan', '27': 'Minnesota', '28': 'Mississippi',
    '29': 'Missouri', '30': 'Montana', '31': 'Nebraska', '32': 'Nevada', '33': 'New Hampshire', '34': 'New Jersey',
    '35': 'New Mexico', '36': 'New York', '37': 'North Carolina', '38': 'North Dakota', '39': 'Ohio', '40': 'Oklahoma',
    '41': 'Oregon', '42': 'Pennsylvania', '44': 'Rhode Island', '45': 'South Carolina', '46': 'South Dakota',
    '47': 'Tennessee', '48': 'Texas', '49': 'Utah', '50': 'Vermont', '51': 'Virginia', '53': 'Washington',
    '54': 'West Virginia', '55': 'Wisconsin', '56': 'Wyoming', '11': 'District of Columbia'
}


def plot_laplace_results_old(dataset_name):
    with open(f'{RESULTS_DIR}/{dataset_name}-laplace-results-pc-split.pt', 'rb') as f:
        cl_results, pc_cl_results, laplace_results, train_sizes, lambdas = torch.load(f,
                                                                                      map_location=torch.device('cpu'))

    cl_losses = np.array([cl_results[train_size] for train_size in train_sizes])
    pc_cl_losses = np.array([pc_cl_results[train_size] for train_size in train_sizes])

    for laplace_lambda in lambdas:
        laplace_losses = np.array([laplace_results[train_size, laplace_lambda] for train_size in train_sizes])
        plt.plot(train_sizes, laplace_losses, label=f'P-c CL, Laplacian $\\lambda={laplace_lambda:g}$', marker='x')

    plt.plot(train_sizes, cl_losses, label='Conditional logit', ls='--', marker='o')
    plt.plot(train_sizes, pc_cl_losses, label='Per-chooser CL', ls='-.', marker='s')

    plt.xlabel('Proportion of train data')
    plt.ylabel('Mean test NLL')
    plt.legend()
    # plt.show()
    plt.savefig(f'plots/laplace-{dataset_name}-pc-split.pdf', bbox_inches='tight')
    plt.close()

    # dataset = Dataset(dataset_name)
    # graph = dataset.chooser_graph
    # nodes = np.array(graph.nodes)
    # m = len(graph.edges)
    # n = len(graph.nodes)
    #
    # print(m, len(graph.nodes))
    # edges = np.array(list(graph.edges()))
    #
    # for seed in seeds:
    #     thetas = pc_cl_results[seed, 0.75][1]['thetas']
    #     print(torch.norm(thetas, dim=1, p=2))
    #
    #     for feat in range(thetas.size(1)):
    #
    #         x = np.concatenate((thetas[:, feat][edges[:, 0]], thetas[:, feat][edges[:, 1]])),
    #         y = np.concatenate((thetas[:, feat][edges[:, 1]], thetas[:, feat][edges[:, 0]]))
    #
    #         r, p = assortativity(graph, thetas[:, feat])
    #         if p < 0.001:
    #             print(f'{dataset.item_feature_names[feat]}: r = {r:.2f}, p = {p:.2g}')
    #             # plt.scatter(x, y)
    #             # plt.title(f'{dataset.item_feature_names[feat]}: r = {r:.2f}, p = {p:.2g}')
    #             #
    #             # plt.show()
    #
    #
    #     distances = torch.norm(thetas[edges[:, 0]] - thetas[edges[:, 1]], dim=1, p=2).numpy()
    #     # rand_edges = edges[:, :]
    #     # np.random.shuffle(rand_edges[:, 0])
    #     rand_edges = np.random.randint(0, len(nodes), (m, 2))
    #     same_idx = rand_edges[:, 0] == rand_edges[:, 1]
    #     while np.count_nonzero(same_idx) > 0:
    #         print(np.sum(same_idx))
    #         rand_edges[same_idx] = np.random.randint(0, len(nodes), (np.sum(same_idx), 2))
    #         same_idx = rand_edges[:, 0] == rand_edges[:, 1]
    #
    #     rand_distances = torch.norm(thetas[rand_edges[:, 0]] - thetas[rand_edges[:, 1]], dim=1, p=2).numpy()
    #
    #     print(distances.mean(), rand_distances.mean())
    #
    #     # plt.hist(distances, bins=100, alpha=0.5, label='real edges')
    #     # plt.hist(rand_distances, bins=100, alpha=0.5, label='rand edges')
    #     # plt.legend()
    #     # plt.title('')
    #     # plt.show()
    #
    #     degrees = np.log(1+np.array([val for node, val in graph.degree(range(n))]))
    #     theta_norms = torch.norm(thetas, dim=1, p=2).numpy()
    #     print(degrees.shape, theta_norms.shape)
    #     plt.scatter(degrees, theta_norms, alpha=0.2)
    #     plt.xlabel('log(1 + degree)')
    #     plt.ylabel('Preference vector 2-norm')
    #     r, p = stats.pearsonr(degrees, theta_norms)
    #     plt.title(f'r = {r:.2g}, p = {p:.2g}')
    #     plt.show()
    #
    #     # total_diff = 0
    #     #
    #     # for i, j in graph.edges:
    #     #     total_diff += torch.norm(thetas[i] - thetas[j]) ** 2
    #     #
    #     # total_rand_diff = 0
    #     # random_samples = 0
    #     # while random_samples < m:
    #     #     i = np.random.choice(nodes)
    #     #     j = np.random.choice(nodes)
    #     #
    #     #     if i != j and not graph.has_edge(i, j):
    #     #         random_samples += 1
    #     #         total_rand_diff += torch.norm(thetas[i] - thetas[j]) ** 2
    #     #
    #     # print(total_diff / m, total_rand_diff / m)


def assortativity(edges, property):
    return stats.pearsonr(np.concatenate((property[edges[:, 0]], property[edges[:, 1]])),
                          np.concatenate((property[edges[:, 1]], property[edges[:, 0]])))


def examine_homophily(dataset_name):
    dataset = Dataset(dataset_name)
    graph = dataset.chooser_graph

    locs = np.argmax(dataset.item_df.drop(['dist_from_last', 'is_repeat'], axis=1).values, axis=1) + 1
    locs[0] = 0

    choice_df = dataset.choice_df.loc[dataset.choice_df['chosen'] == 1, :]
    choice_df['loc'] = locs[choice_df['item'].values]

    unique_locs = choice_df['loc'].unique()

    visit_rates = np.zeros((dataset.n_choosers, len(unique_locs)))

    for i in range(dataset.samples):
        visit_rates[dataset.choosers[i].item(), choice_df['loc'].iloc[i]] += 1

    visit_rates /= visit_rates.sum(1, keepdims=True)

    for loc in range(visit_rates.shape[1]):
        r, p = assortativity(graph, visit_rates[:, loc])
        if p < 0.001:
            print(f'loc_{loc}: r = {r:.2f}, p = {p:.2g}')


def plot_laplace_results(dataset_name, item_id=False):
    fname = f'{RESULTS_DIR}/{dataset_name}-laplace-results.pt'

    if item_id:
        fname = fname.replace('.pt', '-item-id.pt')

    with open(fname, 'rb') as f:
        cl_results, pc_cl_results, laplace_results, train_sizes, laplace_lambdas, l2_lambdas, lrs = torch.load(f,
                                                                                                               map_location=torch.device(
                                                                                                                   'cpu'))

    for index, name in ((3, 'NLL'), (4, 'Accuracy')):
        cl_vals = np.array(
            [min((cl_results[train_size, 0, lr] for lr in lrs), key=lambda x: x[1][-1])[index] for train_size in
             train_sizes])
        pc_cl_vals = np.array([min((pc_cl_results[train_size, l2_lambda, lr] for lr in lrs for l2_lambda in l2_lambdas),
                                   key=lambda x: x[1][-1])[index] for train_size in train_sizes])

        for laplace_lambda in laplace_lambdas:
            laplace_vals = np.array([min((laplace_results[train_size, laplace_lambda, l2_lambda, lr] for lr in lrs for
                                          l2_lambda in l2_lambdas), key=lambda x: x[1][-1])[index] for train_size in
                                     train_sizes])
            plt.plot(train_sizes, laplace_vals, label=f'P-c CL, Laplacian $\\lambda={laplace_lambda:g}$', marker='x')

        plt.plot(train_sizes, cl_vals, label='Conditional logit', ls='--', marker='o')
        plt.plot(train_sizes, pc_cl_vals, label='Per-chooser CL', ls='-.', marker='s')

        plt.xlabel('Proportion of train data')
        plt.ylabel(f'Mean test {name}')
        plt.legend()
        # plt.show()
        plt.savefig(f'plots/laplace-{dataset_name}{"-item-id" if item_id else ""}-{name.lower()}.pdf',
                    bbox_inches='tight')
        plt.close()


def plot_pc_param_distribution(dataset_name):
    node_features = ['log_in_degree', 'log_shared_neighbors', 'log_forward_weight', 'log_reverse_weight',
                     'send_recency', 'receive_recency', 'reverse_recency', 'forward_recency']

    fname = f'{RESULTS_DIR}/{dataset_name}-laplace-results.pt'

    print(dataset_name)

    with open(fname, 'rb') as f:
        cl_results, pc_cl_results, laplace_results, train_sizes, laplace_lambdas, l2_lambdas, lrs = torch.load(f,
                                                                                                               map_location=torch.device(
                                                                                                                   'cpu'))

    train_frac = train_sizes[len(train_sizes) // 2]
    assert train_frac == 0.5

    loss, acc, state_dict, l2_lambda, lr = pc_cl_results['best_model', train_frac]

    cl_theta = cl_results['best_model', train_frac][2]['theta'].numpy()

    global_theta = state_dict['global_theta'].numpy()
    thetas = state_dict['thetas'].numpy()

    fig, axes = plt.subplots(2, 4, figsize=(12, 5))

    for i, feature in enumerate(node_features):
        row = i // 4
        col = i % 4

        axes[row, col].set_title(feature)
        axes[row, col].hist(thetas[:, i] + global_theta[i], bins=40)
        # axes[row, col].axvline(cl_theta[i], color='red', ls='dashed')

        if col == 0:
            axes[row, col].set_ylabel('Node Count')

        if row == 1:
            axes[row, col].set_xlabel('Preference Coef.')

    plt.subplots_adjust(hspace=0.3)

    plt.suptitle(dataset_name)

    plt.savefig(f'plots/{dataset_name}-pref-dsn.pdf', bbox_inches='tight')
    plt.close()


def network_assortativity(dataset_name):
    node_features = ['log_in_degree', 'log_shared_neighbors', 'log_forward_weight', 'log_reverse_weight',
                     'send_recency', 'receive_recency', 'reverse_recency', 'forward_recency']

    fname = f'{RESULTS_DIR}/{dataset_name}-laplace-results.pt'

    print(dataset_name)

    with open(fname, 'rb') as f:
        cl_results, pc_cl_results, laplace_results, train_sizes, laplace_lambdas, l2_lambdas, lrs = torch.load(f,
                                                                                                               map_location=torch.device(
                                                                                                                   'cpu'))

    train_frac = train_sizes[len(train_sizes) // 2]
    assert train_frac == 0.5

    loss, acc, state_dict, l2_lambda, lr = pc_cl_results['best_model', train_frac]

    thetas = state_dict['thetas'].numpy()

    dataset = Dataset(dataset_name)
    unique_edges = np.unique(dataset.timestamped_edges[:, :2], axis=0)

    print('out-degree assort.:',
          assortativity(unique_edges, np.bincount(unique_edges[:, 0], minlength=dataset.n_choosers)))
    print('in-degree assort.:',
          assortativity(unique_edges, np.bincount(unique_edges[:, 1], minlength=dataset.n_choosers)))

    for i, feature in enumerate(node_features):
        print(feature, assortativity(unique_edges, thetas[:, i]))
    G = nx.DiGraph()
    G.add_edges_from(dataset.timestamped_edges[:, :2])

    eigen_centrality = nx.eigenvector_centrality_numpy(G)
    centralities = np.array([eigen_centrality[i] for i in range(dataset.n_choosers)])
    in_degrees = np.array([G.in_degree[i] for i in range(dataset.n_choosers)])
    out_degrees = np.array([G.out_degree[i] for i in range(dataset.n_choosers)])

    # for x_vals, x_name in ((in_degrees, 'In-degree'), (out_degrees, 'Out-degree'), (centralities, 'Centrality')):
    #     fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    #
    #     for i, feature in enumerate(node_features):
    #         row = i // 4
    #         col = i % 4
    #
    #         axes[row, col].scatter(x_vals, thetas[:, i], alpha=0.5, marker='.')
    #         axes[row, col].set_ylabel(f'{node_features[i]} coef.')
    #
    #         slope, intercept, r_value, p_value, std_err = stats.linregress(in_degrees, thetas[:, i])
    #
    #         # axes[row, col].plot(x_vals, slope * x_vals + intercept, color='black')
    #         axes[row, col].text(0.95, 0.95, f'r={r_value:.2f}', horizontalalignment='right', verticalalignment='top', transform=axes[row, col].transAxes)
    #
    #         if row == 1:
    #             axes[row, col].set_xlabel(x_name)
    #
    #     plt.subplots_adjust(wspace=0.35)
    #     plt.savefig(f'plots/network_corr/{dataset_name}-{x_name}-corr.pdf', bbox_inches='tight')

    # print(np.array([eigen_centrality[i] for i in range(dataset.n_choosers)]))


def network_prefs_cluster(dataset_name):
    fname = f'{RESULTS_DIR}/{dataset_name}-laplace-results.pt'

    with open(fname, 'rb') as f:
        cl_results, pc_cl_results, laplace_results, train_sizes, laplace_lambdas, l2_lambdas, lrs = torch.load(f,
                                                                                                               map_location=torch.device(
                                                                                                                   'cpu'))

    train_frac = train_sizes[len(train_sizes) // 2]
    assert train_frac == 0.5

    loss, acc, state_dict, l2_lambda, lr = pc_cl_results['best_model', train_frac]

    thetas = state_dict['thetas'].numpy()

    tsne = TSNE(n_components=2, random_state=1, perplexity=2)
    projected = tsne.fit_transform(thetas)

    for k in range(2, 5):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(thetas)

        plt.scatter(projected[:, 0], projected[:, 1], c=kmeans.labels_)
        plt.text(0.95, 0.95, f'Sil={silhouette_score(thetas, kmeans.labels_):.2f}', horizontalalignment='right',
                 verticalalignment='top',
                 transform=plt.gca().transAxes)

        plt.title(f'{dataset_name}, k={k}')
        plt.show()


def plot_masked_laplacian_results(dataset, item_id=False):
    fname = f'{RESULTS_DIR}/{dataset}-masked-laplace-results{"-item-id" if item_id else ""}.pt'

    with open(fname, 'rb') as f:
        cl_results, pc_cl_results, laplace_results, mask_fracs, laplace_lambdas, l2_lambdas, lrs = torch.load(f,
                                                                                                              map_location=torch.device(
                                                                                                                  'cpu'))

    for index, name in ((3, 'Masked NLL'), (4, 'Masked Accuracy'), (5, 'Unmasked NLL'), (6, 'Unmasked Accuracy')):
        cl_vals = np.array(
            [min((cl_results[mask_frac, l2_lambda, lr] for lr in lrs for l2_lambda in l2_lambdas),
                 key=lambda x: x[1][-1])[index] for mask_frac in
             mask_fracs])
        pc_cl_vals = np.array([min((pc_cl_results[mask_frac, l2_lambda, lr] for lr in lrs for l2_lambda in l2_lambdas),
                                   key=lambda x: x[1][-1])[index] for mask_frac in mask_fracs])

        for laplace_lambda in laplace_lambdas:
            laplace_vals = np.array([min((laplace_results[mask_frac, laplace_lambda, l2_lambda, lr] for lr in lrs for
                                          l2_lambda in l2_lambdas), key=lambda x: x[1][-1])[index] for mask_frac in
                                     mask_fracs])
            plt.plot(mask_fracs, laplace_vals, label=f'P-c CL, Laplacian $\\lambda={laplace_lambda:g}$', marker='x')

        plt.plot(mask_fracs, cl_vals, label='Conditional logit', ls='--', marker='o')
        plt.plot(mask_fracs, pc_cl_vals, label='Per-chooser CL', ls='-.', marker='s')

        plt.xlabel('Fraction of masked nodes')
        plt.ylabel(f'Mean test {name}')
        plt.legend()
        # plt.show()
        plt.savefig(f'plots/masked-laplace-{dataset}-{name}.pdf', bbox_inches='tight')
        plt.close()


def plot_semi_supervised_results(dataset, prox_threshold=None):
    prox_thresh_str = f'-prox-threshold-{prox_threshold}' if prox_threshold is not None else ''
    fname = f'{RESULTS_DIR}/{dataset}-masked-laplace-results-per-item{prox_thresh_str}.pt'


    propagation_fname = f'{RESULTS_DIR}/{dataset}-prop-results-per-item-prox-threshold-{prox_threshold}.pt'
    with open(propagation_fname, 'rb') as f:
        propagation_results = torch.load(f)

    #forgot to save seeds
    # seeds = range(8)

    choice_model = 'logit' if dataset == 'app-install' else 'CL'

    alphas = [0.1, 0.25, 0.5, 0.75, 1]
    colors = ["#001559", "#9ec90c", "#ff8593", "#8c3b00"]

    with open(fname, 'rb') as f:
        cl_results, pc_cl_results, gcn_results, laplace_results, _, train_fracs, laplace_lambdas, l2_lambdas, lrs, chooser_splits, all_idxs, seeds = torch.load(
            f, map_location=torch.device('cpu'))

    # Make dicts of trials with best final validation loss, for each mask frac/seed/laplace lambda
    best_cl_results = {
        (train_frac, seed): min((cl_results[train_frac, l2_lambda, lr, seed] for lr in lrs for l2_lambda in l2_lambdas),
                                key=lambda x: x[1][-1]) for train_frac in train_fracs for seed in seeds}
    best_pc_cl_results = {(train_frac, seed): min(
        (pc_cl_results[train_frac, l2_lambda, lr, seed] for lr in lrs for l2_lambda in l2_lambdas),
        key=lambda x: x[1][-1]) for train_frac in train_fracs for seed in seeds}
    best_gcn_results = {(train_frac, seed): min(
        (gcn_results[train_frac, l2_lambda, lr, seed] for lr in lrs for l2_lambda in l2_lambdas),
        key=lambda x: x[1][-1]) for train_frac in train_fracs for seed in seeds}
    best_laplace_results = {(train_frac, seed): min(
        (laplace_results[train_frac, laplace_lambda, l2_lambda, lr, seed] for lr in lrs for l2_lambda in l2_lambdas for laplace_lambda in laplace_lambdas),
        key=lambda x: x[1][-1]) for train_frac in train_fracs for seed in seeds}

    # Pick alpha by validation MRR
    best_prop_results = {(train_frac, seed): min(
        (propagation_results[train_frac, seed][alpha] for alpha in alphas),
        key=lambda x: x[3]) for train_frac in train_fracs for seed in seeds}

    for index, name in ((3, 'Test NLL'), (4, 'Test Accuracy'), (5, 'Test MRR')):
        plt.figure(figsize=(2.3, 2.1))

        cl_means = np.array(
            [np.mean([best_cl_results[train_frac, seed][index] for seed in seeds]) for train_frac in train_fracs])
        cl_std_errs = np.array(
            [np.std([best_cl_results[train_frac, seed][index] for seed in seeds]) / np.sqrt(len(seeds)) for train_frac in train_fracs])

        gcn_means = np.array(
            [np.mean([best_gcn_results[train_frac, seed][index] for seed in seeds]) for train_frac in train_fracs])
        gcn_std_errs = np.array(
            [np.std([best_gcn_results[train_frac, seed][index] for seed in seeds]) / np.sqrt(len(seeds)) for train_frac in train_fracs])

        # for laplace_lambda in laplace_lambdas[3:]:
        laplace_means = np.array(
            [np.mean([best_laplace_results[train_frac, seed][index] for seed in seeds]) for
             train_frac in train_fracs])
        laplace_std_errs = np.array(
            [np.std([best_laplace_results[train_frac, seed][index] for seed in seeds]) / np.sqrt(len(seeds)) for
             train_frac in train_fracs])

        if index != 3:
            prop_means = np.array([np.mean([best_prop_results[train_frac, seed][index - 4] for seed in seeds])
                                   for train_frac in train_fracs])
            # print(name, pc_cl_means)
            prop_std_errs = np.array(
                [np.std([best_prop_results[train_frac, seed][index - 4] for seed in seeds])
                 for train_frac in train_fracs]) / np.sqrt(len(seeds))

            plt.errorbar(train_fracs, prop_means, prop_std_errs, label=f'propgation', marker='s', ls='dotted',
                         color=colors[3])

        plt.errorbar(train_fracs, cl_means, cl_std_errs, label=choice_model, ls='--', marker='o', color=colors[0])
        plt.errorbar(train_fracs, gcn_means, gcn_std_errs, label='GCN', ls='-.', marker='^', color=colors[1])
        plt.errorbar(train_fracs, laplace_means, laplace_std_errs, label=f'Laplace {choice_model}', marker='x', color=colors[2])

        if name == 'Test NLL':
            print(dataset, 'Laplace likelihood improvement factor:', (np.exp(-laplace_means) / np.exp(-cl_means)))

        elif name == 'Test MRR':
            print(dataset, 'Laplace MRR improvement %:', (cl_means - laplace_means) / cl_means)


        plt.xlabel('Train fraction')
        plt.ylabel(f'Mean {name}')

        if name == 'Test NLL':
            plt.title(dataset)
        else:
            plt.title(' ')

        if dataset == 'app-install' and name == 'Test NLL':
            plt.text(0.63, 4.42, 'logit', color=colors[0])
            plt.text(0.63, 4.57, 'GCN', color=colors[1])
            plt.text(0.22, 4.31, 'Laplacian logit', color=colors[2])
        if dataset == 'app-usage':
            plt.ylabel(None)
            if name == 'Test NLL':
                plt.ylim(1.7, 1.85)

            elif name == 'Test MRR':
                plt.text(0.45, 0.255, 'propagation', color=colors[3])

        # plt.show()
        plt.savefig(f'plots/semi-supervised-{dataset}-{name.lower().replace(" ", "-")}{prox_thresh_str}.pdf',
                    bbox_inches='tight', pad_inches=0)
        plt.close()


def examine_app_results():
    fname = f'{RESULTS_DIR}/app-install-masked-laplace-results-per-item.pt'

    with open(fname, 'rb') as f:
        cl_results, pc_cl_results, laplace_results, train_fracs, laplace_lambdas, l2_lambdas, lrs, chooser_splits, all_idxs, seeds = torch.load(
            f, map_location=torch.device('cpu'))
    #
    data = Dataset('app-install')

    graph = data.chooser_graph

    choices = data.choices.numpy()
    choosers = data.choosers.numpy()

    popularity = np.bincount(choices).astype(float) / data.n_choosers

    installed = np.zeros((data.n_choosers, data.n_items))
    installed[choosers, choices] = 1

    popularities = []
    friend_fracs = []

    for sample in range(data.samples):

        app_popularity = popularity[choices[sample]]
        friends = list(graph.neighbors(choosers[sample]))

        if len(friends) == 0:
            continue
        friend_frac = installed[friends, choices[sample]].sum() / len(friends) if len(friends) > 0 else 0

        popularities.append(app_popularity)
        friend_fracs.append(friend_frac)

    # heatmap, xedges, yedges = np.histogram2d(popularities, friend_fracs, bins=100)
    #
    # print(heatmap)
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.figure(figsize=(4, 3))
    plt.scatter(popularities, friend_fracs, alpha=0.05, linewidth=0)

    plt.xlabel('Global Install Fraction')
    plt.ylabel('Friend Install Fraction')

    # plt.imshow(heatmap, extent=extent, origin='lower', cmap=plt.cm.Reds)
    plt.plot([0.01, 1], [0.01, 1], ls='dashed')

    f = np.poly1d(np.polyfit(popularities, friend_fracs, 1))
    plt.plot(np.logspace(-2, 0, 100), f(np.logspace(-2, 0, 100)))

    plt.yscale('symlog', linthresh=0.01)
    plt.xscale('log')

    plt.savefig('plots/friend-installed-frac.pdf', bbox_inches='tight')

    plt.close()

    # install_counts = np.bincount(choices).astype(float)
    #
    # fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), sharey='row')
    #
    # for col in range(3):
    #
    #     diff_by_friend_frac = defaultdict(float)
    #     counts_by_friend_frac = defaultdict(int)
    #
    #     for app in np.unique(choices):
    #
    #         for chooser in choosers:
    #             friends = list(graph.neighbors(chooser))
    #             friends_installed = installed[friends, app].sum()
    #
    #             expected = (install_counts[app] - friends_installed) / (data.n_choosers - friends_installed)
    #             friend_frac = friends_installed / len(friends) if len(friends) > 0 else 0
    #
    #             diff_by_friend_frac[friend_frac] += installed[chooser, app] - expected
    #             counts_by_friend_frac[friend_frac] += 1
    #
    #     x = sorted(diff_by_friend_frac.keys())
    #     y = [diff_by_friend_frac[x_i] / counts_by_friend_frac[x_i] for x_i in x]
    #
    #     axes[col].scatter(x, y, alpha=0.5, linewidth=0)
    #
    #     nbins = 5
    #
    #     means, bin_edges, _ = binned_statistic(x, y, bins=nbins, statistic='mean')
    #     stds, _, _ = binned_statistic(x, y, bins=nbins, statistic=lambda arr: np.std(arr) / np.sqrt(len(arr)))
    #
    #     bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    #     axes[col].plot(bin_mids, means, color='black', marker='x')
    #     axes[col].fill_between(bin_mids, means - stds, means + stds, alpha=0.3, color='black')
    #
    #     axes[col].hlines(0, 0, 1, color='grey', ls='dashed')
    #     # plt.yscale('log')
    #
    #     if col == 0:
    #         print('swapping')
    #         graph = nx.double_edge_swap(graph, nswap=50000, max_tries=1e6, seed=0)
    #         print('done swapping')
    #     elif col == 1:
    #         n_choose_2 = data.n_choosers * (data.n_choosers + 1) / 2
    #         graph = nx.erdos_renyi_graph(n=data.n_choosers, p=len(graph.edges) / n_choose_2, seed=0)
    #
    #     axes[col].set_xlabel('Friend install frac.')
    #     axes[col].set_xticks([0, 0.5, 1])
    #     axes[col].set_xticklabels(['0', '0.5', '1'])
    #
    # axes[0].set_ylabel(r'$\Pr($install$) - \mathrm{E}[ \Pr($install$)]$')
    #
    # axes[0].set_title('Real network')
    # axes[1].set_title('Reconfigured')
    # axes[2].set_title('Erdős-Rényi')
    #
    # plt.subplots_adjust(wspace=0)
    #
    # plt.savefig('plots/install_rates.pdf', bbox_inches='tight')
    #
    # plt.close()

    # best_params = {(train_frac, laplace_lambda, seed): min(((lr, l2_lambda) for lr in lrs for l2_lambda in l2_lambdas),
    #                                        key=lambda x: laplace_results[train_frac, laplace_lambda, x[1], x[0], seed][1][-1]) for train_frac in
    #                train_fracs for seed in seeds for laplace_lambda in laplace_lambdas}
    #
    # print('\n'.join(map(str, list(best_params.values()))))
    #
    # train_frac = 0.5
    # seed = 1
    #
    # best_cl_results = {
    #     (train_frac, seed): min((cl_results[train_frac, l2_lambda, lr, seed] for lr in lrs for l2_lambda in l2_lambdas),
    #                             key=lambda x: x[1][-1]) for train_frac in train_fracs for seed in seeds}
    #
    # for laplace_lambda in laplace_lambdas:
    #     for lr in lrs:
    #         for l2_lambda in l2_lambdas:
    #
    #             train_losses, val_losses, _, test_loss, _ = laplace_results[train_frac, laplace_lambda, l2_lambda, lr, seed]
    #             x = range(len(train_losses))
    #             plt.plot(x, train_losses, label='train')
    #             plt.plot(x, val_losses, label='val', ls='dashed')
    #             plt.hlines(test_loss, 0, len(train_losses), label='test', ls='dotted')
    #             plt.hlines(best_cl_results[train_frac, seed][3], 0, len(train_losses), label='logit')
    #             plt.ylabel('loss')
    #             plt.xlabel('epoch')
    #             plt.legend()
    #             plt.title(f'laplace: {laplace_lambda}, lr: {lr}, l2: {l2_lambda}')
    #             plt.show()

    # for seed in seeds:
    #     for train_frac in train_fracs:
    #         for l2_lambda in l2_lambdas:
    #             for laplace_lambda in laplace_lambdas:
    #                 for lr in lrs:
    #
    #
    #
    # state_dict = pc_cl_results['best_model', 0.1, 0][2]
    #
    # train, val, test = all_idxs[0.1, 0]
    # train_choosers, val_choosers, test_choosers = chooser_splits[0.1, 0]
    #
    # print(len(train), len(val), len(test))
    # print(len(train_choosers), len(val_choosers), len(test_choosers))
    #
    # print(state_dict)
    #
    # print(list(state_dict['utilities'][train_choosers][0]))
    # print(state_dict['utilities'][val_choosers])
    # print(state_dict['utilities'][test_choosers])


def plot_election_results(dataset, graph):
    nation_best_acc = 0.6152717324300186

    fname = f'{RESULTS_DIR}/{dataset}-masked-laplace-results-{graph}.pt'

    with open(fname, 'rb') as f:
        cl_results, pc_cl_results, gcn_results, laplace_results, _, train_fracs, laplace_lambdas, l2_lambdas, lrs, all_masked_counties, all_test_idxs, seeds = torch.load(
            f, map_location=torch.device('cpu'))

    propagation_fname = f'{RESULTS_DIR}/{dataset}-prop-results-{graph}.pt'
    with open(propagation_fname, 'rb') as f:
        propagation_results = torch.load(f)

    if dataset == 'election-2016':
        dataset = 'us-election-2016'

    alphas = [0.1, 0.25, 0.5, 0.75, 1]

    # Make dicts of trials with best final validation loss, for each mask frac/seed/laplace lambda
    best_cl_results = {
        (train_frac, seed): min((cl_results[train_frac, l2_lambda, lr, seed] for lr in lrs for l2_lambda in l2_lambdas),
                                key=lambda x: x[1][-1]) for train_frac in train_fracs for seed in seeds}
    best_pc_cl_results = {(train_frac, seed): min(
        (pc_cl_results[train_frac, l2_lambda, lr, seed] for lr in lrs for l2_lambda in l2_lambdas),
        key=lambda x: x[1][-1]) for train_frac in train_fracs for seed in seeds}

    best_gcn_results = {(train_frac, seed): min(
        (gcn_results[train_frac, l2_lambda, lr, seed] for lr in lrs for l2_lambda in l2_lambdas),
        key=lambda x: x[1][-1]) for train_frac in train_fracs for seed in seeds}

    best_laplace_results = {(train_frac, seed): min(
        (laplace_results[train_frac, laplace_lambda, l2_lambda, lr, seed] for lr in lrs for l2_lambda in l2_lambdas for laplace_lambda in laplace_lambdas),
        key=lambda x: x[1][-1]) for train_frac in train_fracs for seed in seeds}

    # Pick alpha by validation MRR
    best_prop_results = {(train_frac, seed): min(
        (propagation_results[train_frac, seed][alpha] for alpha in alphas),
        key=lambda x: x[3]) for train_frac in train_fracs for seed in seeds}

    colors = ["#001559", "#9ec90c", "#ff8593", "#8c3b00"]
    for index, name in ((3, 'Test NLL'), (4, 'Test Accuracy'), (5, 'Test MRR')):
        plt.figure(figsize=(2.3, 2.1))

        cl_means = np.array(
            [np.mean([best_cl_results[train_frac, seed][index] for seed in seeds]) for train_frac in train_fracs])
        cl_std_errs = np.array(
            [np.std([best_cl_results[train_frac, seed][index] for seed in seeds]) for train_frac in train_fracs]) / np.sqrt(len(seeds))

        pc_cl_means = np.array(
            [np.mean([best_pc_cl_results[train_frac, seed][index] for seed in seeds]) for train_frac in train_fracs])
        # print(name, pc_cl_means)
        pc_cl_stds = np.array(
            [np.std([best_pc_cl_results[train_frac, seed][index] for seed in seeds]) for train_frac in train_fracs])

        gcn_means = np.array(
            [np.mean([best_gcn_results[train_frac, seed][index] for seed in seeds]) for train_frac in train_fracs])
        # print(name, pc_cl_means)
        gcn_std_errs = np.array(
            [np.std([best_gcn_results[train_frac, seed][index] for seed in seeds]) for train_frac in train_fracs]) / np.sqrt(len(seeds))

        # for laplace_lambda in laplace_lambdas:
        laplace_means = np.array(
            [np.mean([best_laplace_results[train_frac, seed][index] for seed in seeds]) for
             train_frac in train_fracs])
        laplace_std_errs = np.array(
            [np.std([best_laplace_results[train_frac, seed][index] for seed in seeds]) for
             train_frac in train_fracs]) / np.sqrt(len(seeds))

        if index != 3:
            prop_means = np.array([np.mean([best_prop_results[train_frac, seed][index - 4] for seed in seeds])
                                   for train_frac in train_fracs])
            # print(name, pc_cl_means)
            prop_std_errs = np.array(
                [np.std([best_prop_results[train_frac, seed][index - 4] for seed in seeds])
                 for train_frac in train_fracs]) / np.sqrt(len(seeds))

            plt.errorbar(train_fracs, prop_means, prop_std_errs, label=f'propgation', marker='s', ls='dotted', color=colors[3])

            if index == 4 and dataset == 'us-election-2016':
                plt.hlines(nation_best_acc, 0.1, 0.8, color='black')

        if name == 'Test NLL':
            print(dataset, 'Laplace likelihood improvement factor:', (np.exp(-laplace_means) / np.exp(-cl_means)))
            print(dataset, 'Laplace log-like improvement %:', (cl_means - laplace_means) / cl_means)
        elif name == 'Test MRR':
            print(dataset, 'Laplace MRR improvement %:', (cl_means - laplace_means) / cl_means)


        # for alpha in alphas:
        #     prop_means = np.array(
        #         [np.mean([prop_results[train_frac, seed, alpha][index - 3] for seed in seeds]) for train_frac in
        #          train_fracs])
        #
        #     plt.plot(train_fracs, prop_means, label=f'P-c MNL, propagation $\\alpha={alpha:g}$', marker='+',
        #              ls='dotted')

        plt.errorbar(train_fracs, cl_means, cl_std_errs, label='MNL', ls='--', marker='o', color=colors[0])
        # plt.plot(train_fracs, pc_cl_means, label='Per-chooser MNL', ls='-.', marker='s')
        plt.errorbar(train_fracs, gcn_means, gcn_std_errs, label='GCN', ls='-.', marker='^', color=colors[1])
        plt.errorbar(train_fracs, laplace_means, laplace_std_errs, label=f'Laplacian MNL', marker='x', color=colors[2])

        plt.xlabel('Train fraction')
        # plt.ylabel(f'Mean {name}')
        # plt.yscale('log')

        if dataset == 'us-election-2016' and graph == 'social':
            # if name == 'Test NLL':
            #     # plt.text(0.15, 0.882, 'MNL', color=colors[0])
            #     # plt.text(0.12, 0.848, 'Laplacian MNL', color=colors[2])
            #     # plt.text(0.62, 0.8685, 'GCN', color=colors[1])
            # elif name == 'Test MRR':
            #     plt.text(0.45, 0.136, 'propagation', color=colors[3])
            ...
        elif dataset == 'ca-election-2016':
            if name == 'Test NLL':
                plt.ylim(0.643, 0.646)
            # plt.yticks([0.643, 0.644, 0.645, 0.646])


        # else:
        #     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.show()
        if name == 'Test NLL':
            plt.title(dataset)
        else:
            plt.title(' ')
        plt.savefig(f'plots/masked-laplace-{dataset}-{graph}-{name.replace(" ", "_").lower()}.pdf',
                    bbox_inches='tight',  pad_inches=0)

        plt.close()

    print('best loss', np.array(
        [np.mean([pc_cl_results['best_model', mask_frac, seed][0] for seed in seeds]) for mask_frac in train_fracs]))

    # for index, name in ((0, 'Train NLL'), (1, 'Val NLL')):
    #     cl_means = np.array(
    #         [np.mean([best_cl_results[train_frac, seed][index][-1] for seed in seeds]) for train_frac in train_fracs])
    #     cl_stds = np.array(
    #         [np.std([best_cl_results[train_frac, seed][index][-1] for seed in seeds]) for train_frac in train_fracs])
    #
    #     pc_cl_means = np.array(
    #         [np.mean([best_pc_cl_results[train_frac, seed][index][-1] for seed in seeds]) for train_frac in train_fracs])
    #     pc_cl_stds = np.array(
    #         [np.std([best_pc_cl_results[train_frac, seed][index][-1] for seed in seeds]) for train_frac in train_fracs])
    #
    #     for laplace_lambda in laplace_lambdas:
    #         laplace_means = np.array(
    #             [np.mean([best_laplace_results[train_frac, laplace_lambda, seed][index][-1] for seed in seeds]) for train_frac
    #              in train_fracs])
    #         laplace_stds = np.array(
    #             [np.std([best_laplace_results[train_frac, laplace_lambda, seed][index][-1] for seed in seeds]) for train_frac
    #              in train_fracs])
    #         plt.plot(train_fracs, laplace_means, label=f'P-c CL, Laplacian $\\lambda={laplace_lambda:g}$', marker='x')
    #
    #     plt.plot(train_fracs, cl_means, label='Conditional logit', ls='--', marker='o')
    #     plt.plot(train_fracs, pc_cl_means, label='Per-chooser CL', ls='-.', marker='s')
    #
    #     plt.xlabel('Fraction of masked counties')
    #     plt.ylabel(f'Mean {name}')
    #     if index == 3:
    #         plt.legend()
    #     # plt.show()
    #     plt.savefig(f'plots/masked-laplace-election-2016-{graph}-{name.replace(" ", "_").lower()}.pdf', bbox_inches='tight')
    #     plt.close()


def election_2016_predictions(data, graph, choice_sets, choice_set_sizes, chooser_features, choosers):

    fname = f'{RESULTS_DIR}/election-2016-masked-laplace-results-{graph}.pt'

    with open(fname, 'rb') as f:
        mnl_results, pc_mnl_results, gcn_results, laplace_results, propagation_results, train_fracs, laplace_lambdas, \
            l2_lambdas, lrs, all_counties, all_idxs, seeds = torch.load(f, map_location=torch.device('cpu'))

    pc_mnl_init_args = (data.n_chooser_features, data.n_items, data.n_choosers)

    n_voters = np.zeros(data.n_choosers)
    df = data.choice_df[data.choice_df.chosen == 1]
    df['count'] = data.choice_counts.int()[df['sample'].values]

    for fips, votes in df.groupby('chooser')['count']:
        n_voters[data.reindex_choosers_map[fips]] = votes.sum()

    n_voters = n_voters[choosers]
    total_voters = n_voters.sum()

    train_frac = train_fracs[-1]

    cands = data.item_df['name']

    preds = {cand: [] for cand in cands}

    chooser_id_to_fips = {v: k for k, v in data.reindex_choosers_map.items()}
    state_totals = {state: {cand: [0 for _ in seeds] for cand in cands} for state in state_fips_map.values()}

    for seed in seeds:
        laplace_state_dict = laplace_results['best_model', train_frac, seed][3]

        model = PerChooserMultinomialLogit(*pc_mnl_init_args)

        model.load_state_dict(laplace_state_dict)

        log_probs = model(choice_sets, choice_set_sizes, chooser_features, choosers).detach().numpy()

        pred_votes = np.exp(log_probs) * n_voters[:, None]

        pred_totals = torch.zeros(len(cands)).double()
        for sample in range(len(choice_sets)):
            pred_totals[choice_sets[sample, :choice_set_sizes[sample]]] += pred_votes[sample, :choice_set_sizes[sample]]

        # print('pred totals:', pred_totals)
        # print('vs straight sum:', pred_votes.sum(0)),
        # pred_totals = pred_votes.sum(0)

        for i, name in enumerate(cands):
            preds[name].append(pred_totals[i] / total_voters * 100)

        for j, chooser_id in enumerate(choosers.numpy()):
            county_fips = str(chooser_id_to_fips[chooser_id]).zfill(5)
            state_fips = county_fips[:2]

            for i, cand_idx in enumerate(choice_sets[j, :choice_set_sizes[j]]):
                state_totals[state_fips_map[state_fips]][cands[cand_idx.item()]][seed] += pred_votes[j, i]

    return cands, preds, total_voters, state_totals


def election_full_ballot_prediction(graph):
    _, _, national_vote_totals, national_possible_vote_totals, state_vote_totals = get_2016_election_vote_pcts()


    data = Dataset('election-2016')
    cands = data.item_df['name']

    clinton_idx = list(cands).index('Hillary Clinton')
    trump_idx = list(cands).index('Donald Trump')
    johnson_idx = list(cands).index('Gary Johnson')
    stein_idx = list(cands).index('Jill Stein')
    mcmullin_idx = list(cands).index('Evan McMullin')
    none_idx = list(cands).index(' None of these candidates')


    # print(data.chooser_df)
    # print(data.reindex_choosers_map)
    # choice_sets = torch.arange(data.n_items).repeat(data.n_choosers, 1)

    choice_set = torch.zeros(data.n_items).long()
    choice_set[0] = clinton_idx
    choice_set[1] = trump_idx
    choice_set[2] = johnson_idx
    choice_set[3] = stein_idx
    choice_set[4] = mcmullin_idx
    choice_sets = choice_set.repeat(data.n_choosers, 1)

    print(choice_sets)

    choice_set_sizes = torch.full((data.n_choosers,), 5)
    choosers = torch.from_numpy(data.chooser_df.index.values)
    chooser_features = torch.nan_to_num(
        (torch.from_numpy(data.chooser_df.values) - data.chooser_feature_means) / data.chooser_feature_stds)

    cands, preds, total_voters, state_totals = election_2016_predictions(data, graph, choice_sets, choice_set_sizes, chooser_features, choosers)

    print('========== SCENARIO 1 ============')

    print_state_outcome_changes(cands, preds, state_totals, state_vote_totals)

    #
    # #
    # # plt.figure(figsize=(12, 8))
    # # plt.barh(np.arange(len(cands)), [np.mean(preds[cand]) for cand in cands],
    # #          xerr=[np.std(preds[cand]) for cand in cands], zorder=-1, label='Predicted Full-Ballot %')
    # # plt.yticks(np.arange(len(cands)), cands)
    # # plt.scatter([national_vote_totals[cand] / total_voters * 100 for cand in cands], np.arange(len(cands)), zorder=1,
    # #             label='National %')
    # # plt.scatter([national_vote_totals[cand] / national_possible_vote_totals[cand] * 100 for cand in cands],
    # #             np.arange(len(cands)), zorder=1, marker='x', label='Extrapolated %')
    # #
    # # plt.xlabel('National Vote Percent')
    # # plt.legend()
    # #
    # # plt.savefig('plots/election-pred-2016.pdf', bbox_inches='tight')
    # # plt.close()
    #

    print('========== SCENARIO 2 ============')
    # Scenario 2
    choice_set = torch.zeros(data.n_items).long()
    choice_set[0] = clinton_idx
    choice_set[1] = trump_idx

    choice_sets = choice_set.repeat(data.n_choosers, 1)
    choice_set_sizes = torch.full((data.n_choosers,), 2)

    cands, preds, total_voters, state_totals = election_2016_predictions(data, graph, choice_sets, choice_set_sizes, chooser_features, choosers)
    print_state_outcome_changes(cands, preds, state_totals, state_vote_totals)


    # Get real choice sets for each county

    clinton_choice_idxs = data.choice_sets[torch.arange(data.samples), data.choices] == clinton_idx

    choice_sets = torch.cat((data.choice_sets[clinton_choice_idxs],
                             torch.full((clinton_choice_idxs.sum(), len(cands) - data.choice_sets.size(1)), 0)), 1)
    choice_set_sizes = data.choice_set_sizes[clinton_choice_idxs]
    chooser_features = data.chooser_features[clinton_choice_idxs]
    choosers = data.choosers[clinton_choice_idxs]

    # print(data.reindex_item_id_map)
    # print(data.reindex_choosers_map)
    # # Make sure this lines up with chooser features
    # permute_idx = torch.argsort(choosers)

    # assert (data.choosers[clinton_choice_idxs] == choosers[permute_idx]).all()
    #
    # chooser_features = chooser_features[permute_idx]
    # choosers = choosers[permute_idx]

    # choice_set_sizes = choice_set_sizes[permute_idx]

    print('========== REAL BALLOTS ============')

    cands, preds, total_voters, state_totals = election_2016_predictions(data, graph, choice_sets, choice_set_sizes,
                                                                         chooser_features, choosers)
    print_state_outcome_changes(cands, preds, state_totals, state_vote_totals)

    for sample in range(len(choice_sets)):
        if none_idx not in choice_sets[sample]:
            choice_sets[sample, choice_set_sizes[sample]] = none_idx
            choice_set_sizes[sample] += 1


    # print(choice_set_sizes)
    #
    print('========== SCENARIO 3 ============')

    cands, preds, total_voters, state_totals = election_2016_predictions(data, graph, choice_sets, choice_set_sizes,
                                                                         chooser_features, choosers)
    print_state_outcome_changes(cands, preds, state_totals, state_vote_totals)



def print_state_outcome_changes(cands, preds, state_totals, state_vote_totals):
    for state in state_totals:
        pred_cand_order = sorted(cands, key=lambda x: np.mean(state_totals[state][x]), reverse=True)
        real_cand_order = sorted(cands, key=lambda x: np.mean(state_vote_totals[state, x]), reverse=True)

        show_k = min(3, len(pred_cand_order))

        if pred_cand_order[0] != real_cand_order[0]:
            print(state)
            print(f'Pred:', ', '.join(
                f'{pred_cand_order[i]} ({int(np.mean(state_totals[state][pred_cand_order[i]]))})' for i in range(show_k)))
            print(f'Real:', ', '.join(
                f'{real_cand_order[i]} ({int(np.mean(state_vote_totals[state, real_cand_order[i]]))})' for i in
                range(3)))
            print()

    print('Popular vote estimates:')
    print('mean', {cand: round(np.mean(preds[cand]), 1) for cand in cands})
    print('CI', {cand: round(np.std(preds[cand]) / np.sqrt(len(preds[cand])) * 1.96, 1) for cand in cands})



def get_2016_election_vote_pcts():
    data = Dataset('election-2016')

    vote_counts = dict()
    vote_pcts = dict()
    national_vote_totals = defaultdict(int)
    state_vote_totals = defaultdict(int)
    national_possible_vote_totals = defaultdict(int)

    for fips, cand_idx in data.choice_df[['chooser', 'item']].drop_duplicates().values:
        vote_pcts[fips, data.item_df.loc[cand_idx].values[0]] = 0
        vote_counts[fips, data.item_df.loc[cand_idx].values[0]] = 0


    df = data.choice_df[data.choice_df.chosen == 1]
    df['count'] = data.choice_counts.int()[df['sample'].values]
    df['candidate'] = data.item_df.loc[df.item.values].values
    df['county_total'] = df.groupby('chooser')['count'].transform('sum')
    df['pct'] = df['count'] / df['county_total']

    print(df.groupby('chooser')['count'].max().sum() / df['count'].sum())

    for _, fips, _, _, count, candidate, county_total, pct in df.values:
        vote_pcts[fips, candidate] = pct
        vote_counts[fips, candidate] = count
        national_vote_totals[candidate] += count
        national_possible_vote_totals[candidate] += county_total

        state_fips = str(fips).zfill(5)[:2]
        state_vote_totals[state_fips_map[state_fips], candidate] += count

    return vote_pcts, vote_counts, national_vote_totals, national_possible_vote_totals, state_vote_totals


def plot_2016_election_maps():
    data = Dataset('election-2016')

    county_shapefile = f'{DATA_DIR}/usa-2016-election/USA_Counties-shp/USA_Counties.shp'
    state_shapefile = f'{DATA_DIR}/usa-2016-election/cb_2018_us_state_500k/cb_2018_us_state_500k.shp'

    county_shapes = Reader(county_shapefile)
    state_shapes = Reader(state_shapefile)

    vote_pcts, vote_counts, _, _, _ = get_2016_election_vote_pcts()

    custom_cmaps = {'Hillary Clinton': plt.cm.Blues, 'Donald Trump': plt.cm.Reds}

    for candidate in data.item_df.name:
        if candidate != 'Evan McMullin':
            continue

        print('Making plot for', candidate)
        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.LambertConformal()))

        cmap = custom_cmaps[candidate] if candidate in custom_cmaps else plt.cm.Greens

        # norm = plt.Normalize(vmin=0, vmax=max(vote_counts[key] / (vote_counts[key] + vote_counts[key[0], 'Hillary Clinton'] + vote_counts[key[0], 'Donald Trump']) for key in vote_pcts.keys() if key[1] == candidate))
        # norm = plt.Normalize(vmin=0, vmax=max(vote_pcts[key] for key in vote_pcts.keys() if key[1] == candidate))
        norm = plt.Normalize(vmin=0, vmax=max(1 - vote_pcts[key[0], 'Hillary Clinton'] - vote_pcts[key[0], 'Donald Trump'] for key in vote_pcts.keys() if key[1] == candidate))

        for county in county_shapes.records():

            fips = int(county.attributes['FIPS'])

            if (fips, candidate) in vote_pcts:
                # incl_trump_clinton_total = vote_counts[fips, candidate] + vote_counts[fips, 'Hillary Clinton'] + \
                #                            vote_counts[fips, 'Donald Trump']

                ax.add_geometries([county.geometry], ccrs.PlateCarree(),
                                  facecolor=cmap(norm(vote_pcts[fips, candidate])), edgecolor='none')
                # ax.add_geometries([county.geometry], ccrs.PlateCarree(),
                #                   facecolor=cmap(norm(vote_counts[fips, candidate] / incl_trump_clinton_total)), edgecolor='none')

                # ax.add_geometries([county.geometry], ccrs.PlateCarree(),
                #                   facecolor=cmap(norm(1 - vote_pcts[fips, 'Hillary Clinton'] - vote_pcts[fips, 'Donald Trump'] )),
                #                   edgecolor='none')

                # print(fips, vote_counts[fips, candidate] / incl_trump_clinton_total, vote_pcts[fips, candidate])
            else:
                ax.add_geometries([county.geometry], ccrs.PlateCarree(), facecolor='black')

        for state in state_shapes.geometries():
            edgecolor = 'black'
            ax.add_geometries([state], ccrs.PlateCarree(), facecolor='none', edgecolor=edgecolor, linewidth=0.2)

        print('Saving....')
        ax.set_extent([-125, -66.5, 20, 50], ccrs.Geodetic())

        # https://stackoverflow.com/a/47790537/6866505
        divider = make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size='5%', pad=0.1, axes_class=plt.Axes)
        fig.add_axes(ax_cb)
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_cb, label='Vote Share')

        # plt.savefig(f'{candidate}-vote-share.png', bbox_inches='tight', dpi=1000)
        plt.savefig(f'third-party-vote-share.png', bbox_inches='tight', dpi=1000)

        plt.close()

    # county_feature = ShapelyFeature(Reader(county_shapefile).geometries(), ccrs.PlateCarree())
    # ax.add_feature(county_feature, edgecolor='black', facecolor='white', linewidth=0.2)
    #
    # county_feature = ShapelyFeature(Reader(county_shapefile).geometries(), ccrs.PlateCarree())
    # ax.add_feature(county_feature, edgecolor='black', facecolor='None', linewidth=0.4)
    #


def plot_network_convergence():
    fig, axes = plt.subplots(1, 2, sharey='row', figsize=(6, 2.5))

    cmap = plt.get_cmap('plasma')
    colors = cmap(np.linspace(0, 1, 6))
    widths = [1, 1.5, 2, 2.5, 3, 3.5]

    labels = [0, '10^{-6}', '10^{-5}', '10^{-4}', '10^{-3}', '10^{-2}']
    for col, laplace in enumerate(['', '-no']):

        with open(f'{RESULTS_DIR}/network-convergence{laplace}-laplace-reg.pt', 'rb') as f:
            results, items, n, k, p, trial_range, sample_range, chooser_range, lambda_range = torch.load(f)

        # lambda_range = lambda_range[2:]

        # colors = ["#98b7c3", "#8e53c2", "#9ed063", "#c55780", "#627246", "#c66d3d", "#4f3750"]

        for i, l in enumerate(lambda_range):
            # for trial in trial_range:
            #     plt.plot(sample_range, [y for x, y in results[trial, l]], marker='x', color=colors[i])

            mean = np.mean([results[trial, l] for trial in trial_range], axis=0)
            std = np.std([results[trial, l] for trial in trial_range], axis=0)
            axes[col].errorbar(sample_range, mean, std, label=f'$\lambda={labels[i]}$', marker='.', color=colors[i], lw=widths[i], markersize=10, ls='dashed' if l == 0 else 'solid', alpha=0.8)
            # axes[col].fill_between(sample_range, mean - std, mean + std, alpha=0.2, color=colors[i], lw=widths[i])

        axes[col].set_xscale('log')
        # axes[col].set_yscale('log')

        axes[col].set_xlabel('Samples per Chooser')

        axes[col].set_yscale('log')

    axes[0].set_title(f'Laplacian Reg.')
    axes[1].set_title(f'No Regularization')

    axes[1].legend()
    axes[0].set_ylabel('Mean Squared Error')

    plt.savefig(f'plots/network-convergence-small-alpha.pdf', bbox_inches='tight')
    plt.close()


def plot_timing():
    model_names = {
        choice_models.Logit: 'CL/MNL',
        choice_models.PerChooserLogit: 'Laplacian CL/MNL',
        choice_models.GCNEmbeddingLogit: 'GCN',
        choice_models.ConditionalLogit: 'CL/MNL',
        choice_models.PerChooserConditionalLogit: 'Laplacian CL/MNL',
        choice_models.GCNEmbeddingConditionalLogit: 'GCN',
        choice_models.MultinomialLogit: 'CL/MNL',
        choice_models.PerChooserMultinomialLogit: 'Laplacian CL/MNL',
        choice_models.GCNMultinomialLogit: 'GCN'
    }

    with open(f'results/timing-results.pt', 'rb') as f:
        datasets, models, times = torch.load(f)

    for dataset in datasets:
        print(f'\\textsc{{{dataset}}} &', ' & '.join(str(f'${np.mean(times[dataset][model]):.2f} \pm {np.std(times[dataset][model]) / np.sqrt(len(times[dataset][model])):.2f}$') for model in models[dataset]), '\\')

    # print(times)


def examine_top_apps():
    # write_app_graph_file('app-install', 10)


    data = Dataset('app-install')
    fname = f'{RESULTS_DIR}/app-install-masked-laplace-results-per-item-prox-threshold-10.pt'

    propagation_fname = f'{RESULTS_DIR}/app-install-prop-results-per-item-prox-threshold-10.pt'
    with open(propagation_fname, 'rb') as f:
        propagation_results = torch.load(f)

    with open(fname, 'rb') as f:
        cl_results, pc_cl_results, gcn_results, laplace_results, _, train_fracs, laplace_lambdas, l2_lambdas, lrs, chooser_splits, all_idxs, seeds = torch.load(
            f, map_location=torch.device('cpu'))

    train_frac = train_fracs[-1]

    mean_cl_utils = np.zeros(data.n_items)

    mean_laplace_utils = np.zeros((data.n_choosers, data.n_items))

    for seed in seeds:
        _, _, _, cl_state_dict, _, _, _ = cl_results['best_model', train_frac, seed]
        mean_cl_utils += cl_state_dict['utilities'].numpy()


        _, _, _, laplace_state_dict, _, _, _, _ = laplace_results['best_model', train_frac, seed]
        mean_laplace_utils += laplace_state_dict['global_utilities'].numpy() + laplace_state_dict['utilities'].numpy()

    mean_cl_utils /= len(seeds)

    cl_order = np.argsort(mean_cl_utils)

    app_name_map = {app_id: name for name, app_id in data.reindex_item_id_map.items()}
    app_names = np.array([app_name_map[i] for i in range(data.n_items)])

    print('Table 5')
    for i in cl_order[-20:][::-1]:
        print(f'{{{app_names[i]}}} & {mean_cl_utils[i]:.2f}) \\\\')

    print(mean_laplace_utils)

    mean_mean_laplace_utils = mean_laplace_utils.mean(0)
    laplace_order = np.argsort(mean_mean_laplace_utils)

    top_10_laplace_count = np.zeros(data.n_items)

    myspace_idx = list(app_names).index('com.myspace.android')
    facebook_idx = list(app_names).index('com.facebook.katana')

    myspace_participants = set()
    facebook_participants = set()

    for chooser in range(data.n_choosers):
        order = np.argsort(mean_laplace_utils[chooser])
        top_10_laplace_count[order[-10:]] += 1

        if facebook_idx in order[-10:]:
            facebook_participants.add(chooser)

        if myspace_idx in order[-10:]:
            myspace_participants.add(chooser)

    top_10_order = np.argsort(top_10_laplace_count)

    print('Table 6')
    for i in reversed(top_10_order[-30:]):
        print(f'{{{app_names[i]}}} & {int(top_10_laplace_count[i])} \\\\')

    print('FACEBOOK', facebook_participants)
    print('MYSPACE', myspace_participants)
    print(len(facebook_participants), len(myspace_participants), len(facebook_participants.intersection(myspace_participants)),
          len(facebook_participants.union(myspace_participants)))

    overlap = facebook_participants.intersection(myspace_participants)
    # Without overlap
    # facebook_participants = facebook_participants - overlap
    # myspace_participants = myspace_participants - overlap

    facebook_participant_subgraph = nx.subgraph(data.chooser_graph, facebook_participants)
    myspace_participant_subgraph = nx.subgraph(data.chooser_graph, myspace_participants)
    social_net_participant_subgraph = nx.subgraph(data.chooser_graph, facebook_participants.union(myspace_participants))

    facebook_edges = len(facebook_participant_subgraph.edges)
    myspace_edges = len(myspace_participant_subgraph.edges)
    all_edges = len(social_net_participant_subgraph.edges)
    between_edges = all_edges - facebook_edges - myspace_edges

    print('FACEBOOK ')
    print('Fraction of facebook edges', facebook_edges / ((len(facebook_participants) * (len(facebook_participants) -1))/2))
    print('Fraction of myspace edges', between_edges / (len(facebook_participants)*len(myspace_participants)))

    print('MYSPACE')
    print('Fraction of facebook edges', between_edges / (len(myspace_participants) * len(facebook_participants)))
    print('Fraction of myspace edges', myspace_edges / ((len(myspace_participants) * (len(myspace_participants)-1)) / 2))

    # print(facebook_degree, myspace_degree)
    # print('Facebook participants:', f'Pr of edge to a myspace participant: {(between_edges / len(facebook_participants)) / len(myspace_participants)}',
    #       f'Pr of edge to a facebook participant: {(facebook_edges / len(facebook_participants)) / len(facebook_participants)}')
    #
    # print('Myspace participants:',
    #       f'Pr of edge to a myspace participant: {(between_edges / len(myspace_participants)) / len(myspace_participants)}',
    #       f'Pr of edge to a facebook participant: {(myspace_edges / len(myspace_participants)) / len(facebook_participants)}')



if __name__ == '__main__':
    yelp_datasets = [
        'yelp-austin', 'yelp-atlanta', 'yelp-boulder', 'yelp-columbus', 'yelp-vancouver',
        'yelp-orlando', 'yelp-boston',  # 'yelp-portland',
    ]

    network_datasets = [
        'college-msg', 'email-eu-core', 'bitcoin-alpha', 'bitcoin-otc', 'email-enron-core', 'email-w3c-core',
        'high-school-contact-2011', 'high-school-contact-2012'
    ]

    checkin_datasets = [
        'brightkite-denver', 'brightkite-london', 'brightkite-san-francisco', 'brightkite-tokyo',
        'gowalla-honolulu', 'gowalla-dallas', 'gowalla-boston', 'brightkite-los-angeles', 'gowalla-seattle',
        'gowalla-oslo'
    ]

    trivago_datasets = [
        'trivago-london', 'trivago-nyc', 'trivago-tokyo', 'trivago-paris'
    ]

    friends_and_family_datasets = [
         'app-usage', 'app-install'
    ]

    os.makedirs('plots', exist_ok=True)
    plot_timing()

    plot_election_results('election-2016', 'social')
    plot_election_results('ca-election-2016', 'adjacency')
    plot_election_results('ca-election-2020', 'adjacency')
    #
    election_full_ballot_prediction('social')

    for dataset in friends_and_family_datasets:
        for prox_thresh in [10]:

            plot_semi_supervised_results(dataset, prox_thresh)

    examine_top_apps()
    plot_network_convergence()
    plot_2016_election_maps()

