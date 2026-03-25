from copy import deepcopy
from copy import deepcopy
from typing import TypedDict

import optuna
import torch
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted, validate_data
from sksurv.base import SurvivalAnalysisMixin
from sksurv.util import check_array_survival
from torch import nn
from torch_survival.losses import weibull_neg_log_likelihood, weibull_neg_log_likelihood_original, weibull_survival_time
from torch_survival.metrics import concordance_index
from torch_survival.utils import merge_configs


class SparsityLayer(nn.Module):
    def __init__(self, n_dists):
        super().__init__()
        print('Initializing sparsity layer')
        self.weight = nn.Parameter(torch.Tensor(n_dists))
        torch.nn.init.uniform_(self.weight, 0, 1)

    def normalize_weights(self):
        with torch.no_grad():
            self.weight /= self.weight.sum()

    def forward(self, alphas):
        return alphas * (self.weight / self.weight.sum())


class WeibullNetwork(nn.Module):
    def __init__(self, n_inputs, n_dists, sparse=False, original=False):
        super().__init__()
        if sparse and not original:
            raise ValueError('Using sparse=True requires original=True')
        self.sparse = sparse
        self.original = original
        # As described in paper section 4.1 Network Configuration
        self.shared_network = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self.params_network = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
        )
        self.shape_layer = nn.Linear(8, n_dists)
        self.scale_layer = nn.Linear(8, n_dists)
        self.weights_network = None
        if original or n_dists > 1:
            self.weights_network = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, 8),
                nn.BatchNorm1d(8),
                nn.ReLU(inplace=True),
                nn.Linear(8, n_dists),
            )
        self.sparsity_layer = None
        if sparse:
            self.sparsity_layer = SparsityLayer(n_dists)

    def normalize_weights(self):
        if self.sparsity_layer:
            self.sparsity_layer.normalize_weights()

    def forward(self, x):
        x = self.shared_network(x)
        x_params = self.params_network(x)
        shape = torch.nn.functional.elu(self.shape_layer(x_params)) + 2
        scale = torch.nn.functional.elu(self.scale_layer(x_params)) + 1 + 1e-4
        if not self.weights_network:  # single Weibull distribution
            return torch.cat((shape, scale), dim=-1)
        else:  # multiple Weibull distributions
            weights = self.weights_network(x)
            if self.original:
                weights = torch.softmax(weights, dim=-1)
            if self.sparse:
                weights = self.sparsity_layer(weights)
                weights /= weights.sum(dim=-1, keepdim=True)
            return torch.cat((weights, shape, scale), dim=-1)


class WeibullSearchSpace(TypedDict):
    #: Number of Weibull distributions for survival time estimation
    n_dists: tuple[int, int] | int


class WeibullBase(SurvivalAnalysisMixin, BaseEstimator):
    #: Default hyperparameter search space
    default_search_space: WeibullSearchSpace = {
        'n_dists': (1, 10),
    }

    def __init__(self, search_space: WeibullSearchSpace | None = None, n_dists=2, n_epochs=500, lamb=1e-4, sparse=False, original=False, random_state=None,
                 device=None):
        self.search_space = deepcopy(self.default_search_space)
        if search_space:
            self.search_space = merge_configs(self.search_space, search_space)
        self.n_dists = n_dists
        self.n_epochs = n_epochs
        self.lamb = lamb
        self.sparse = sparse
        self.original = original
        self.random_state = random_state
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _optimize(self, trial: optuna.Trial, X, y_event, y_time):
        # Do 5-fold cross validation
        scores = []
        for train_idx, test_idx in StratifiedKFold(n_splits=5).split(X, y_event.cpu()):
            model = self._train(trial, X[train_idx], y_event[train_idx], y_time[train_idx])
            model.eval()
            params = model(X[test_idx]).squeeze(dim=-1)
            times = self.time_horizon_ * weibull_survival_time(params, softmax=not self.original)
            c_index = concordance_index(times, y_event[test_idx], y_time[test_idx], mode='time')
            scores.append(c_index)
        return - sum(scores) / len(scores)

    def _train(self, X, y_event, y_time):
        # Set up model, optimizer, and scheduler
        n_inputs, n_outputs = X.shape[-1], 1
        model = WeibullNetwork(n_inputs, self.n_dists, self.sparse, self.original)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Train and return model
        model.to(self.device)
        for i in range(self.n_epochs):
            optimizer.zero_grad()
            params = model(X).squeeze(-1)
            if self.original:  # original implementation
                loss = weibull_neg_log_likelihood_original(params, y_event, y_time)
            else:  # optimized implementation with fused softmax
                loss = weibull_neg_log_likelihood(params, y_event, y_time)
            if self.sparse:
                # Regularize the weights to encourage sparsity
                loss += self.lamb * torch.sum(torch.sqrt(torch.abs(model.sparsity_layer.weight)))
            loss.backward()
            optimizer.step()
            model.normalize_weights()
        return model

    def fit(self, X, y):
        """ Fit the model to the given survival data.

        Parameters
        ----------
        X: array-like, shape = (n_samples, n_features)
            Data matrix.
        y: structured array, shape = (n_samples,)
            A structured array with two fields. The first field is a boolean where ``True`` indicates an event and
            ``False`` indicates right-censoring. The second field is a float with the time of event or time of
            censoring.

        Returns
        -------
        self
        """
        # Validate and extract data
        X, y = validate_data(self, X, y)
        y_event, y_time = check_array_survival(X, y)

        # Convert data to tensors
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_event = torch.as_tensor(y_event, dtype=torch.bool, device=self.device)
        y_time = torch.as_tensor(y_time.copy(), dtype=torch.float32, device=self.device)

        # Seed PyTorch random number generator
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        # Normalize target time
        self.time_horizon_ = y_time.max().item()
        y_time /= self.time_horizon_

        # Optimize hyperparameters
        # optuna.logging.disable_default_handler()
        # warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
        # with OptunaProgressCallback(model_name='DeepWeiSurv', n_trials=10) as callback:
        #     search_space = {'n_dists': list(range(self.search_space['n_dists'][0], self.search_space['n_dists'][1] + 1))}
        #     study = optuna.create_study(sampler=GridSampler(search_space=search_space))
        #     objective = functools.partial(self._optimize, X=X, y_event=y_event, y_time=y_time)
        #     study.optimize(objective, n_trials=10, callbacks=[callback])

        # Train model
        self.model_ = self._train(X, y_event, y_time)
        self.model_.eval()

        return self

    @torch.no_grad()
    def predict(self, X):
        """ Predict survival times.

        The survival time is estimated based on the mean of the mixture of Weibull distributions predicted by the
        neural network.

        Parameters
        ----------
        X: array-like, shape = (n_samples, n_features)
            Data matrix.

        Returns
        -------
        survival_time: array, shape = (n_samples,)
            Predicted survival times.
        """
        check_is_fitted(self)
        X = validate_data(self, X)
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        params = self.model_(X).detach()
        if self.sparse:
            mean_weights = params[:, :self.n_dists].mean(axis=0)
            mask = mean_weights > 0.1  # TODO: weight threshold config
            params = params[:, mask.repeat(3)]
        times = self.time_horizon_ * weibull_survival_time(params, softmax=not self.original)
        return times.cpu().numpy()


class DeepWeiSurv(WeibullBase):
    r""" Implements the DeepWeiSurv model presented by Bennis et al. [1]_.

    Uses a deep neural network trained with the negative log likelihood of the survival time probability distribution,
    as estimated by the parameters of a mixture of Weibull distributions. This implementation tries to stay faithful
    to the original paper, with the following deviations:

    .. [1] A. Bennis, S. Mouysset, and M. Serrurier, “Estimation of Conditional Mixture Weibull Distribution with Right
       Censored Data Using Neural Network for Time-to-Event Analysis,” Lecture Notes in Computer Science. Springer
       International Publishing, pp. 687–698, 2020. doi: 10.1007/978-3-030-47426-3_53. Available:
       http://dx.doi.org/10.1007/978-3-030-47426-3_53
    """

    def __init__(self, n_dists=2, n_epochs=1000, random_state=None, device=None):
        super().__init__(n_dists=n_dists, n_epochs=n_epochs, device=device, sparse=False, original=False,
                         random_state=random_state)


class DPWTE(WeibullBase):
    r""" Implements the DPWTE model presented by Bennis et al. [1]_.

    Uses a deep neural network trained with the negative log likelihood of the survival time probability distribution,
    as estimated by the parameters of a mixture of Weibull distributions. This implementation tries to stay faithful
    to the original paper, with the following deviations:

    .. [1]
    """

    def __init__(self, n_dists_max=10, lamb=1e-4, n_epochs=1000, random_state=None, device=None):
        super().__init__(n_dists=n_dists_max, n_epochs=n_epochs, lamb=lamb, device=device, sparse=True, original=True,
                         random_state=random_state)
