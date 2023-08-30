from abc import ABC

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as nnf
from scipy.stats import stats
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from tqdm import tqdm

from datasets import Dataset

torch.set_num_threads(1)

# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


class ChoiceModel(torch.nn.Module):
    def laplacian_regularization(self, left_nodes, right_nodes, laplace_lambda):
        reg = torch.tensor(0, dtype=torch.float, device=device)
        for param in self.per_chooser_parameters:
            reg += (param[left_nodes] - param[right_nodes]).pow(2).sum()

        return reg * laplace_lambda

    def l2_regularization(self, l2_lambda):
        l2_reg = torch.tensor(0, dtype=torch.float, device=device)
        for param in self.parameters():
            l2_reg += torch.pow(param, 2).sum()

        return l2_lambda * l2_reg

    def accuracy(self, y_hat, y):
        """
        Compute accuracy (fraction of choice set correctly predicted)
        :param y_hat: log(choice probabilities)
        :param y: observed choices
        :return: the accuracy
        """
        return (y_hat.argmax(1).int() == y.int()).float().mean()

    def mean_relative_rank(self, y_hat, y):
        """
        Compute mean rank of correct answer in output sorted by probability
        :param y_hat:
        :param y:
        :return:
        """
        return np.mean(self.relative_ranks(y_hat, y))

    def relative_ranks(self, y_hat, y):
        """
        Compute mean rank of correct answer in output sorted by probability
        :param y_hat:
        :param y:
        :return:
        """
        y_hat = y_hat.squeeze()
        y = y.squeeze()

        choice_set_lengths = np.array((~torch.isinf(y_hat)).sum(1))
        ranks = stats.rankdata(-y_hat.detach().numpy(), method='average', axis=1)[np.arange(len(y)), y] - 1

        return ranks / (choice_set_lengths - 1)

    @property
    def num_params(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)


class ItemIdentityChoiceModel(ChoiceModel, ABC):
    ...


class ItemFeatureChoiceModel(ChoiceModel, ABC):
    ...


class PerChoooserChoiceModel(ChoiceModel, ABC):
    ...


class ChooserFeatureChoiceModel(ChoiceModel, ABC):
    ...


class Logit(ItemIdentityChoiceModel):

    def __init__(self, num_items):
        """
        Initialize an MNL model for inference
        :param num_items: number of unique items
        """
        super().__init__()
        self.num_items = num_items

        self.utilities = torch.nn.Parameter(torch.zeros(self.num_items))

    def forward(self, choice_sets, choice_set_sizes):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_sets: the choice sets
        :param choice_set_sizes: number of items in each choice set
        :return: log(choice probabilities) over every choice set
        """
        batch_size, max_choice_set_len = choice_sets.size()

        # Initialize all utilities to -inf, then set all utilities within choice_set_sizes according to self.utilities
        utilities = torch.full((batch_size, max_choice_set_len), -np.inf, device=device)
        idx = torch.arange(max_choice_set_len, device=device)[None, :] < choice_set_sizes[:, None]
        utilities[idx] = self.utilities[choice_sets[idx]]

        return nnf.log_softmax(utilities, 1)


class PerChooserLogit(ItemIdentityChoiceModel, PerChoooserChoiceModel):
    def __init__(self, num_items, num_choosers):
        super().__init__()
        self.num_items = num_items
        self.num_choosers = num_choosers
        self.global_utilities = torch.nn.Parameter(torch.zeros(self.num_items))
        self.utilities = torch.nn.Parameter(torch.zeros(self.num_choosers, self.num_items))
        self.per_chooser_parameters = [self.utilities]

    def forward(self, choice_sets, choice_set_sizes, choosers):
        batch_size, max_choice_set_len = choice_sets.size()

        utilities = self.utilities[choosers[:, None], choice_sets] + self.global_utilities[choice_sets]

        idx = torch.arange(max_choice_set_len, device=device)[None, :] >= choice_set_sizes[:, None]
        utilities[idx] = -np.inf

        return nnf.log_softmax(utilities, 1)


class ConditionalLogit(ItemFeatureChoiceModel):
    def __init__(self, num_item_feats, num_items=None):
        """
        :param num_item_feats: number of item features
        """
        super().__init__()
        self.num_item_feats = num_item_feats
        self.num_items = num_items
        self.theta = torch.nn.Parameter(torch.zeros(self.num_item_feats))

        if self.num_items is not None:
            self.intercepts = torch.nn.Parameter(torch.zeros(self.num_items))

    def forward(self, choice_set_features, choice_set_sizes, choice_sets=None):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_set_features: the choice sets, with item features
        :param choice_set_sizes: number of items in each choice set
        :param choice_sets: items in each choice set, if intercepts are being used
        :return: log(choice probabilities) over every choice set
        """
        batch_size, max_choice_set_len, num_feats = choice_set_features.size()

        utilities = (self.theta * choice_set_features).sum(-1)
        idx = torch.arange(max_choice_set_len, device=device)[None, :] >= choice_set_sizes[:, None]
        utilities[idx] = -np.inf

        if choice_sets is not None:
            utilities[~idx] += self.intercepts[choice_sets[~idx]]

        return nnf.log_softmax(utilities, 1)


class PerChooserConditionalLogit(ItemFeatureChoiceModel, PerChoooserChoiceModel):
    def __init__(self, num_item_feats, num_choosers, num_items=None):
        """
        :param num_item_feats: number of item features
        :param num_choosers: number of choosers
        """
        super().__init__()
        self.num_choosers = num_choosers
        self.num_item_feats = num_item_feats
        self.num_items = num_items
        self.thetas = torch.nn.Parameter(torch.zeros(self.num_choosers, self.num_item_feats))
        self.global_theta = torch.nn.Parameter(torch.zeros(self.num_item_feats))
        self.per_chooser_parameters = [self.thetas]

        if self.num_items is not None:
            self.intercepts = torch.nn.Parameter(torch.zeros(self.num_choosers, self.num_items))
            self.per_chooser_parameters.append(self.intercepts)
            self.global_intercept = torch.nn.Parameter(torch.zeros(self.num_items))

    def forward(self, choice_set_features, choice_set_sizes, choosers, choice_sets=None):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_set_features: the choice sets, with item features
        :param choice_set_sizes: number of items in each choice set
        :return: log(choice probabilities) over every choice set
        """
        batch_size, max_choice_set_len, num_feats = choice_set_features.size()

        utilities = ((self.global_theta + self.thetas[choosers, None, :]) * choice_set_features).sum(-1)
        idx = torch.arange(max_choice_set_len, device=device)[None, :] >= choice_set_sizes[:, None]
        utilities[idx] = -np.inf

        if choice_sets is not None:
            utilities += torch.gather(self.intercepts[choosers], 1, choice_sets)
            utilities[~idx] += self.global_intercept[choice_sets[~idx]]

        return nnf.log_softmax(utilities, 1)


class MultinomialLogit(ChooserFeatureChoiceModel, ItemIdentityChoiceModel):
    def __init__(self, num_chooser_feats, num_items):
        """
        :param num_item_feats: number of chooser features
        """
        super().__init__()
        self.num_chooser_feats = num_chooser_feats
        self.num_items = num_items
        self.coeffs = torch.nn.Parameter(torch.ones(self.num_items, num_chooser_feats))
        self.intercepts = torch.nn.Parameter(torch.zeros(self.num_items))

    def forward(self, choice_sets, choice_set_sizes, chooser_features):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_sets: the choice sets
        :param chooser_features: features of the chooser in each sample
        :return: log(choice probabilities) over every choice set
        """
        batch_size, max_choice_set_len = choice_sets.size()

        utilities = (self.coeffs[choice_sets] * chooser_features[:, None, :]).sum(axis=2) + self.intercepts[choice_sets]

        idx = torch.arange(max_choice_set_len, device=device)[None, :] >= choice_set_sizes[:, None]
        utilities[idx] = -np.inf

        return nnf.log_softmax(utilities, 1)


class PerChooserMultinomialLogit(ChooserFeatureChoiceModel, ItemIdentityChoiceModel, PerChoooserChoiceModel):
    def __init__(self, num_chooser_feats, num_items, num_choosers):
        """
        :param num_item_feats: number of chooser features
        """
        super().__init__()
        self.num_chooser_feats = num_chooser_feats
        self.num_items = num_items
        self.num_choosers = num_choosers
        self.coeffs = torch.nn.Parameter(torch.ones(self.num_items, num_chooser_feats))
        self.intercepts = torch.nn.Parameter(torch.zeros(self.num_choosers, self.num_items))
        self.global_intercept = torch.nn.Parameter(torch.zeros(self.num_items))
        self.per_chooser_parameters = [self.intercepts]

    def forward(self, choice_sets, choice_set_sizes, chooser_features, choosers):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_sets: the choice sets
        :param chooser_features: features of the chooser in each sample
        :return: log(choice probabilities) over every choice set
        """
        batch_size, max_choice_set_len = choice_sets.size()

        utilities = (self.coeffs[choice_sets] * chooser_features[:, None, :]).sum(axis=2) + self.intercepts[choosers[:, None], choice_sets] + self.global_intercept[choice_sets]

        idx = torch.arange(max_choice_set_len, device=device)[None, :] >= choice_set_sizes[:, None]
        utilities[idx] = -np.inf

        return nnf.log_softmax(utilities, 1)


class GCNMultinomialLogit(ChooserFeatureChoiceModel, ItemIdentityChoiceModel):
    def __init__(self, node_features, num_items, normalized_A, hidden_dim, output_dim, dropout_p):
        """
        :param num_item_feats: number of chooser features
        """
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.normalized_A = normalized_A
        self.W_1 = torch.nn.Linear(node_features.size(1), self.hidden_dim)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.W_2 = torch.nn.Linear(self.hidden_dim, self.output_dim)

        # Save some computation by storing the first convolution
        self._cached_first_convolution = self.normalized_A @ node_features
        del node_features

        self.coeffs = torch.nn.Parameter(torch.ones(self.num_items, 2 * output_dim))
        self.intercepts = torch.nn.Parameter(torch.zeros(self.num_items))

    def get_embeddings(self) -> torch.Tensor:
        """
        Returns: GCN node embeddings
        """
        first_layer = self.dropout(torch.tanh(self.W_1(self._cached_first_convolution)))
        second_layer = self.W_2(self.normalized_A @ first_layer)

        return torch.cat((first_layer, second_layer), dim=1)

    def forward(self, choice_sets, choice_set_sizes, choosers):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_sets: the choice sets
        :param chooser_features: features of the chooser in each sample
        :return: log(choice probabilities) over every choice set
        """
        batch_size, max_choice_set_len = choice_sets.size()

        embeddings = self.get_embeddings()[choosers]

        utilities = (self.coeffs[choice_sets] * embeddings[:, None, :]).sum(axis=2) + self.intercepts[choice_sets]

        idx = torch.arange(max_choice_set_len, device=device)[None, :] >= choice_set_sizes[:, None]
        utilities[idx] = -np.inf

        return nnf.log_softmax(utilities, 1)


class GCNEmbeddingLogit(ItemIdentityChoiceModel):
    def __init__(self, num_choosers, num_items, normalized_A, embedding_dim, hidden_dim, output_dim, dropout_p):
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.node_embedding = torch.nn.Parameter(torch.zeros(num_choosers, embedding_dim))

        self.normalized_A = normalized_A
        self.W_1 = torch.nn.Linear(embedding_dim, self.hidden_dim)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.W_2 = torch.nn.Linear(self.hidden_dim, self.output_dim)

        self.coeffs = torch.nn.Parameter(torch.ones(self.num_items, 2 * output_dim))
        self.intercepts = torch.nn.Parameter(torch.zeros(self.num_items))

    def get_embeddings(self) -> torch.Tensor:
        """
        Returns: GCN node embeddings
        """
        first_layer = self.dropout(torch.tanh(self.W_1(self.node_embedding)))
        second_layer = self.W_2(self.normalized_A @ first_layer)

        return torch.cat((first_layer, second_layer), dim=1)

    def forward(self, choice_sets, choice_set_sizes, choosers):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_sets: the choice sets
        :param chooser_features: features of the chooser in each sample
        :return: log(choice probabilities) over every choice set
        """
        batch_size, max_choice_set_len = choice_sets.size()

        embeddings = self.get_embeddings()[choosers]

        utilities = (self.coeffs[choice_sets] * embeddings[:, None, :]).sum(axis=2) + self.intercepts[choice_sets]

        idx = torch.arange(max_choice_set_len, device=device)[None, :] >= choice_set_sizes[:, None]
        utilities[idx] = -np.inf

        return nnf.log_softmax(utilities, 1)


class GCNEmbeddingConditionalLogit(ItemFeatureChoiceModel, ItemIdentityChoiceModel):
    def __init__(self, num_items, num_item_feats, num_choosers, normalized_A, embedding_dim, hidden_dim, output_dim, dropout_p):
        """
        :param num_item_feats: number of item features
        :param num_choosers: number of choosers
        """
        super().__init__()
        self.num_choosers = num_choosers
        self.num_item_feats = num_item_feats
        self.num_items = num_items

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.node_embedding = torch.nn.Parameter(torch.zeros(num_choosers, embedding_dim))

        self.normalized_A = normalized_A
        self.W_1 = torch.nn.Linear(embedding_dim, self.hidden_dim)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.W_2 = torch.nn.Linear(self.hidden_dim, self.output_dim)

        self.theta = torch.nn.Parameter(torch.zeros(self.num_item_feats))
        self.coeffs = torch.nn.Parameter(torch.ones(self.num_items, 2 * output_dim))

        self.intercepts = torch.nn.Parameter(torch.zeros(self.num_items))

    def get_embeddings(self) -> torch.Tensor:
        """
        Returns: GCN node embeddings
        """
        first_layer = self.dropout(torch.tanh(self.W_1(self.node_embedding)))
        second_layer = self.W_2(self.normalized_A @ first_layer)

        return torch.cat((first_layer, second_layer), dim=1)

    def forward(self, choice_set_features, choice_sets, choice_set_sizes, choosers):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_set_features: the choice sets, with item features
        :param choice_set_sizes: number of items in each choice set
        :param choice_sets: items in each choice set, if intercepts are being used
        :return: log(choice probabilities) over every choice set
        """
        batch_size, max_choice_set_len, num_feats = choice_set_features.size()

        embeddings = self.get_embeddings()[choosers]

        utilities = (self.coeffs[choice_sets] * embeddings[:, None, :]).sum(axis=2) + (self.theta * choice_set_features).sum(-1)
        idx = torch.arange(max_choice_set_len, device=device)[None, :] >= choice_set_sizes[:, None]
        utilities[idx] = -np.inf

        utilities[~idx] += self.intercepts[choice_sets[~idx]]

        return nnf.log_softmax(utilities, 1)


def train(model, train_set, val_set, model_arg_idxs, choice_idx, epochs=100, learning_rate=5e-2, l2_lambda=1e-4,
          show_live_loss=False, show_progress=False, laplace_lambda=1, edges=None, sample_count_idx=None, val_increase_break=False, dont_break=False):
    """
    Fit a choice model to data.

    :param model: a ChoiceModel
    :param train_set: a torch.utils.data.dataset.Subset for training
    :param val_set: a torch.utils.data.dataset.Subset for validation
    :param model_arg_idxs: indices of the models args in the dataset
    :param choice_idx: index of the actual choices the dataset
    :param epochs: number of optimization epochs
    :param learning_rate: step size hyperparameter for Rprop
    :param l2_lambda: regularization hyperparameter
    :param show_live_loss: if True, add loss/accuracy to progressbar
    :param show_progress: if true, show progressbar
    """

    torch.set_num_threads(1)

    optimizer = torch.optim.Rprop(model.parameters(), lr=learning_rate)

    progress_bar = tqdm(range(epochs), total=epochs) if show_progress else range(epochs)

    use_laplace = edges is not None
    if use_laplace:
        left_nodes, right_nodes = edges[:, 0].to(device), edges[:, 1].to(device)

    train_losses = []
    val_losses = []
    gradients = []

    train_data = train_set[:]
    val_data = val_set[:]

    model_args = [train_data[idx] for idx in model_arg_idxs]
    choices = train_data[choice_idx]

    if sample_count_idx is not None:
        train_sample_counts = train_data[sample_count_idx]

    for epoch in progress_bar:
        model.train()

        optimizer.zero_grad()
        log_probs = model(*model_args)

        if sample_count_idx is None:
            loss = nnf.nll_loss(log_probs, choices)
        else:
            loss = torch.dot(nnf.nll_loss(log_probs, choices, reduction='none').squeeze(), train_sample_counts) / train_sample_counts.sum()

        train_loss = loss.item()

        loss += model.l2_regularization(l2_lambda)

        if use_laplace:
            loss += model.laplacian_regularization(left_nodes, right_nodes, laplace_lambda)

        loss.backward(retain_graph=True)

        with torch.no_grad():
            gradient = torch.stack([(item.grad ** 2).sum() for item in model.parameters()]).sum().item()

        optimizer.step()

        train_losses.append(train_loss)
        gradients.append(gradient)

        model.eval()
        val_losses.append(_test(model, val_data, model_arg_idxs, choice_idx, sample_count_idx=sample_count_idx)[0])

        if show_progress and show_live_loss:
            progress_bar.set_description(f'Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Grad: {gradients[-1]:.3e}. Epochs')

        if not dont_break:
            # Break if converged, or if val loss increases and we're doing val_increase_break
            if gradient < 10 ** -8 or (epoch > 0 and val_increase_break and val_losses[-1] > val_losses[-2]):
                break

    # if isinstance(model, ConditionalLogit):
    #     std_errs = None
        # if l2_lambda == 0:
        #     fisher_matrix = torch.zeros((model.num_params, model.num_params), device=device)
        #     for sample in tqdm(range(len(choices))):
        #         model.zero_grad()
        #
        #         # Need gradient at each sample, so compute loss at one sample
        #         loss = nnf.nll_loss(model(*[arg[[sample]] for arg in model_args]), choices[[sample]])
        #         loss.backward(retain_graph=True)
        #         gradient = torch.cat([param.grad.flatten() for param in model.parameters()])
        #
        #         # Fisher information matrix is sum of outer products of gradients at each sample
        #         fisher_matrix += torch.outer(gradient, gradient) * (1 if sample_count_idx is None else train_sample_counts[sample])
        #
        #     fisher_matrix /= len(choices) if sample_count_idx is None else train_sample_counts.sum()
        #     fisher_matrix = fisher_matrix.cpu().numpy()
        #     try:
        #         covariance = np.linalg.inv(fisher_matrix)
        #         std_errs = np.sqrt(np.diagonal(covariance) / (len(choices) if sample_count_idx is None else train_sample_counts.sum()))
        #     except np.linalg.LinAlgError:
        #         std_errs = None

        # return train_losses, val_losses, gradients, std_errs

    return train_losses, val_losses, gradients


def _test(model, test_data, model_arg_idxs, choice_idx, sample_count_idx=None):
    model_args = [test_data[idx] for idx in model_arg_idxs]
    choices = test_data[choice_idx]

    if sample_count_idx is not None:
        sample_counts = test_data[sample_count_idx]

    with torch.no_grad():
        logits = model(*model_args)
        if sample_count_idx is None:
            test_loss = nnf.nll_loss(logits, choices).item()
            test_acc = model.accuracy(logits, choices).item()
            test_mrr = model.mean_relative_rank(logits, choices)

        else:
            test_loss = (torch.dot(nnf.nll_loss(logits, choices, reduction='none').squeeze(), sample_counts) / sample_counts.sum()).item()
            test_acc = (torch.dot((logits.argmax(1).int() == choices.int()).float(), sample_counts) / sample_counts.sum()).item()
            test_mrr = (model.relative_ranks(logits, choices) * sample_counts.numpy()).sum() / sample_counts.sum().item()

    return test_loss, test_acc, test_mrr


def test(model, test_set, model_arg_idxs, choice_idx, sample_count_idx=None):
    model.eval()
    return _test(model, test_set[:], model_arg_idxs, choice_idx, sample_count_idx)


def test_choice_propagation(test_set, smoothed, choice_idx, sample_count_idx=None):
    test_set = test_set[:]
    test_choice_sets, test_choice_set_sizes, test_choosers, test_choices = test_set[0], test_set[1], test_set[2], \
                                                                           test_set[choice_idx]

    # Predict choice fractions
    y_hat = np.full(test_choice_sets.shape, -np.inf)
    for sample in range(len(y_hat)):
        y_hat[sample, :test_choice_set_sizes[sample]] = smoothed[
            test_choosers[sample], test_choice_sets[sample, :test_choice_set_sizes[sample]]]

    y_hat = torch.from_numpy(y_hat)
    ranks = stats.rankdata(-y_hat.detach().numpy(), method='average', axis=1)[
                np.arange(len(test_choices)), test_choices] - 1
    rel_ranks = ranks / (test_choice_set_sizes - 1)

    if sample_count_idx is None:
        mrr = rel_ranks.mean().item()
        acc = (y_hat.argmax(1).int() == test_choices.int()).float().mean().item()
    else:
        sample_counts = test_set[sample_count_idx]
        acc = (torch.dot((y_hat.argmax(1).int() == test_choices.int()).float(),
                         sample_counts) / sample_counts.sum()).item()
        mrr = ((rel_ranks * sample_counts.numpy()).sum() / sample_counts.sum()).item()

    return acc, mrr


def choice_propagation(train_set, test_set, graph, choice_idx, n_choosers, n_items, sample_count_idx=None, val_set=None, alphas=None):
    # print('choice propagation')
    if alphas is None:
        alphas = [0.1, 0.25, 0.5, 0.75, 1]

    max_propagation_steps = 256

    train_set = train_set[:]

    # This is the normalized adjacency matrix:
    N = torch.from_numpy(nx.normalized_laplacian_matrix(graph, sorted(graph.nodes)).toarray())
    S = (torch.eye(len(graph.nodes)) - N).float()

    # Make the choice fraction matrix
    choice_sets, choice_set_sizes, choosers, choices = train_set[0], train_set[1], train_set[2], train_set[choice_idx]

    if sample_count_idx is not None:
        sample_counts = train_set[sample_count_idx]

    # Number of times each item (col) appears to each chooser (row)
    train_item_in_set_counts = torch.zeros(n_choosers, n_items)
    train_chosen_counts = torch.zeros(n_choosers, n_items)

    if sample_count_idx is None:
        for sample in range(len(choices)):
            train_item_in_set_counts[choosers[sample], choice_sets[sample, :choice_set_sizes[sample]]] += 1
            train_chosen_counts[choosers[sample], choice_sets[sample, :choice_set_sizes[sample]][choices[sample]]] += 1
    else:
        for sample in range(len(choices)):
            train_item_in_set_counts[choosers[sample], choice_sets[sample, :choice_set_sizes[sample]]] += sample_counts[sample]
            train_chosen_counts[choosers[sample], choice_sets[sample, :choice_set_sizes[sample]][choices[sample]]] += sample_counts[sample]

    # Prevent division by 0
    train_item_in_set_counts[train_item_in_set_counts == 0] = 1

    train_choice_fractions = train_chosen_counts / train_item_in_set_counts

    results = dict()
    for alpha in alphas:
        # Do propagation
        smoothed = train_choice_fractions.clone()
        for k in range(max_propagation_steps):
            before = smoothed.clone()
            smoothed = (alpha * S @ smoothed) + (1 - alpha) * train_choice_fractions

            diff = (before - smoothed).pow(2).sum().item()
            if diff < 10 ** -8:
                break

        results[alpha] = test_choice_propagation(test_set, smoothed, choice_idx, sample_count_idx)

        if val_set is not None:
            results[alpha] += test_choice_propagation(val_set, smoothed, choice_idx, sample_count_idx)

    return results


if __name__ == '__main__':
    data = Dataset('election-2016')

    counties = torch.unique(data.choosers)

    train_counties, val_test_counties = train_test_split(counties, train_size=0.5)
    val_counties, test_counties = train_test_split(val_test_counties, train_size=0.5)

    train_idx = (data.choosers[..., None] == train_counties).any(-1).nonzero().squeeze()
    val_idx = (data.choosers[..., None] == val_counties).any(-1).nonzero().squeeze()
    test_idx = (data.choosers[..., None] == test_counties).any(-1).nonzero().squeeze()

    print(choice_propagation(Subset(data, train_idx), Subset(data, test_idx), data.chooser_graph, 3, data.n_choosers, data.n_items, 5,
                       Subset(data, val_idx)))
