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
from torch_survival.losses import cox_neg_log_likelihood
from torch_survival.metrics import concordance_index
from torch_survival.progress import OptunaProgressCallback
from torch_survival.sample import sample_network, sample_optimizer
from torch_survival.utils import merge_configs


class DeepSurvSearchSpace(TypedDict):
    #: Neural network configuration
    network: NetworkConfig
    #: Optimizer configuration
    optimizer: OptimizerConfig


class DeepSurv(SurvivalAnalysisMixin, BaseEstimator):
    r""" Implements the DeepSurv model presented by Katzman et al. [1]_.

    Uses a deep neural network trained with the Cox negative log partial likelihood to estimate the risk of each
    individual. The network's configuration is tuned using the Sobol solver. This implementation tries to stay faithful
    to the original paper, with the following deviations:

    * Optuna's default TPE sampler is used in favor of the Sobol sampler with 5-fold internal cross-validation instead
      of 3-fold internal cross-validation.
    * The hyperparameter search space is not detailed in the original paper, and the reference implementation is
      underspecified. We thus define our own shared search space in `DeepSurvSearchSpace`.
    * Our implementation does not support or tune :math:`\ell_2` regularization. We found this to be detrimental to
      performance and were unable to fully replicate the described weight regularization.

    .. [1] J. L. Katzman, U. Shaham, A. Cloninger, J. Bates, T. Jiang, and Y. Kluger, “DeepSurv: personalized treatment
       recommender system using a Cox proportional hazards deep neural network,” BMC Med Res Methodol, vol. 18, no. 1,
       Feb. 2018, doi: 10.1186/s12874-018-0482-1. Available: http://dx.doi.org/10.1186/s12874-018-0482-1
    """

    #: Default hyperparameter search space
    default_search_space: DeepSurvSearchSpace = {
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
            'lr': (1e-7, 1e-3),
            'scheduler': 'inverse_time',
            'decay': (0.0, 0.001),
            'momentum': (0.8, 0.95),
        },
    }

    def __init__(self, search_space: DeepSurvSearchSpace | None = None, n_epochs=500, random_state=None, device=None):
        self.search_space = deepcopy(self.default_search_space)
        if search_space:
            self.search_space = merge_configs(self.search_space, search_space)
        self.n_epochs = n_epochs
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
            risks = model(X[test_idx]).squeeze(dim=-1)
            c_index = concordance_index(risks, y_event[test_idx], y_time[test_idx])
            scores.append(c_index)
        return - sum(scores) / len(scores)

    def _train(self, trial: optuna.Trial, X, y_event, y_time):
        # Set up model, optimizer, and scheduler
        n_inputs, n_outputs = X.shape[-1], 1
        model = sample_network(trial, self.search_space['network'], n_inputs, n_outputs)
        optimizer, scheduler = sample_optimizer(trial, self.search_space['optimizer'], model)

        # Pre-sort dataset based on time
        sort_idx = torch.argsort(y_time, descending=True)
        X = X[sort_idx]
        y_event = y_event[sort_idx]
        y_time = y_time[sort_idx]

        # Train and return model
        model.to(self.device)
        for i in range(self.n_epochs):
            optimizer.zero_grad()
            risk = model(X).squeeze(-1)
            loss = cox_neg_log_likelihood(risk, y_event, y_time, sort=False)
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

        # Optimize hyperparameters
        optuna.logging.disable_default_handler()
        warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
        with OptunaProgressCallback(model_name='DeepSurv', n_trials=50) as callback:
            study = optuna.create_study(sampler=TPESampler(seed=self.random_state) if self.random_state else None)
            objective = functools.partial(self._optimize, X=X, y_event=y_event, y_time=y_time)
            study.optimize(objective, n_trials=50, callbacks=[callback])

        # Train final model with best hyperparameters
        self.optuna_params_ = study.best_params
        self.model_ = self._train(study.best_trial, X, y_event, y_time)
        self.model_.eval()

        return self

    @torch.no_grad()
    def predict(self, X):
        """ Predict risk scores.

        The risk score is predicted directly by a neural network. A higher score indicates a higher risk of experiencing
        the event.

        Parameters
        ----------
        X: array-like, shape = (n_samples, n_features)
            Data matrix.

        Returns
        -------
        risk_score: array, shape = (n_samples,)
            Predicted risk scores.
        """
        check_is_fitted(self)
        X = validate_data(self, X)
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return self.model_(X).detach().squeeze(dim=-1).cpu().numpy()

    def get_optuna_params(self):
        return self.optuna_params_
