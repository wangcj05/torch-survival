import functools
import warnings
from copy import deepcopy
from typing import TypedDict

import optuna
import torch
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted, validate_data
from sksurv.base import SurvivalAnalysisMixin
from sksurv.util import check_array_survival

from torch_survival.config import NetworkConfig, OptimizerConfig
from torch_survival.metrics import concordance_index
from torch_survival.progress import OptunaProgressCallback
from torch_survival.sample import sample_network, sample_optimizer
from torch_survival.utils import merge_configs


class RankDeepSurvSearchSpace(TypedDict):
    #: Neural network configuration
    network: NetworkConfig
    #: Optimizer configuration
    optimizer: OptimizerConfig
    #: Number of pairs to process in each batch
    batch_size: tuple[int, int] | int
    #: Weight for ranking loss
    alpha: tuple[float, float] | float


class RankDeepSurv(SurvivalAnalysisMixin, BaseEstimator):
    r""" Implements the RankDeepSurv model presented by Jing et al. [1]_.

    Uses a deep neural network trained with a mean squared error and ranking loss, adapted for censoring, to estimate
    the survival time of each individual. The network's configuration is tuned using a random solver.

    .. [1] B. Jing et al., “A deep survival analysis method based on ranking,” Artificial Intelligence in Medicine, vol.
       98, pp. 1–9, July 2019, doi: 10.1016/j.artmed.2019.06.001. Available:
       http://dx.doi.org/10.1016/j.artmed.2019.06.001
    """

    #: Default hyperparameter search space
    default_search_space: RankDeepSurvSearchSpace = {
        'network': {
            'layers': {
                'max_layers': 4,
                'max_nodes_per_layer': 50,
            },
            'activation': ['relu', 'selu'],
            'dropout': (0.0, 0.5),
        },
        'optimizer': {
            'name': ['sgd', 'adam'],
            'lr': (10e-7, 10e-3),
            'scheduler': 'inverse_time',
            'decay': (0.0, 0.001),
            'momentum': (0.8, 0.95),
        },
        'batch_size': 128,
        'alpha': (0, 1),
    }

    def __init__(self, search_space: RankDeepSurvSearchSpace | None = None, n_epochs=1, random_state=None, device=None):
        self.search_space = deepcopy(self.default_search_space)
        if search_space:
            self.search_space = merge_configs(self.search_space, search_space)
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _optimize(self, trial: optuna.Trial, X, y_event, y_time):
        # Do four-fold cross validation
        scores = []
        for train_idx, test_idx in StratifiedKFold(n_splits=4).split(X, y_event.cpu()):
            model = self._train(trial, X[train_idx], y_event[train_idx], y_time[train_idx])
            model.eval()
            times = model(X[test_idx]).squeeze(dim=-1)
            c_index = concordance_index(-times, y_event[test_idx], y_time[test_idx])
            scores.append(c_index)
        return sum(scores) / len(scores)

    def _train(self, trial, X, y_event, y_time):
        # Set up model, optimizer, and scheduler
        n_inputs, n_outputs = X.shape[-1], 1
        model = sample_network(trial, self.search_space['network'], n_inputs, n_outputs)
        optimizer, scheduler = sample_optimizer(trial, self.search_space['optimizer'], model)

        # Extract data pair indices for ranking loss
        event_i, event_j = y_event.unsqueeze(-2), y_event.unsqueeze(-1)
        time_i, time_j = y_time.unsqueeze(-2), y_time.unsqueeze(-1)
        # The compatibility matrix specifies which pairs (i,j) can be compared accounting for censoring
        comp = event_i & (time_i < time_j)
        pairs = torch.nonzero(comp)  # of shape (n_pairs, 2)

        # Sample batch size and loss parameters
        batch_size = self.search_space['batch_size']
        if not isinstance(batch_size, int):
            batch_size = trial.suggest_int('batch_size', *batch_size)
        alpha = self.search_space['alpha']
        if not isinstance(alpha, float):
            alpha = trial.suggest_float('alpha', *alpha)

        # Train and return model
        model.to(self.device)
        for i in range(self.n_epochs):
            idx = torch.randperm(pairs.shape[0])
            _pairs = pairs[idx]  # reshuffle at each epoch
            for j in range(0, _pairs.shape[0], batch_size):
                optimizer.zero_grad()
                j_idx, i_idx = _pairs[j:j + batch_size, 0], _pairs[j:j + batch_size, 1]
                # mean squared error loss
                e_time_j = model(X[j_idx]).squeeze(-1)
                y_time_j = y_time[j_idx]
                mse_loss = torch.square(y_time_j - e_time_j) * (y_event[j_idx] | (e_time_j <= y_time_j))
                # ranking loss
                e_time_i = model(X[i_idx]).squeeze(-1)
                e_diff_ji = e_time_j - e_time_i
                y_diff_ji = y_time_j - y_time[i_idx]
                rank_loss = torch.square(torch.clamp(y_diff_ji - e_diff_ji, min=0))
                # backpropagation
                loss = torch.mean(alpha * mse_loss + (1 - alpha) * rank_loss)
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
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

        # Optimize hyperparameters using random solver
        optuna.logging.disable_default_handler()
        warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
        with OptunaProgressCallback(model_name='DeepSurv', n_trials=50) as callback:
            study = optuna.create_study(sampler=TPESampler(seed=self.random_state) if self.random_state else None,
                                        direction='maximize')
            objective = functools.partial(self._optimize, X=X, y_event=y_event, y_time=y_time)
            study.optimize(objective, n_trials=50, callbacks=[callback])

        # Train final model with best hyperparameters
        self.optuna_params_ = study.best_params
        self.model_ = self._train(study.best_trial, X, y_event, y_time)
        self.model_.eval()

        return self

    @torch.no_grad()
    def predict(self, X):
        """ Predict survival times.

        The survival time is predicted directly by a neural network.

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
        times = self.model_(X).detach().squeeze(dim=-1).cpu().numpy()
        times = times * self.time_horizon_  # undo normalization
        return times

    def get_optuna_params(self):
        return self.optuna_params_
