import functools
import warnings
from copy import deepcopy
from typing import TypedDict

import numpy as np
import optuna
import torch
import torchtuples as tt
from optuna.samplers import TPESampler
from pycox.evaluation import EvalSurv
from pycox.models import DeepHitSingle
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.validation import check_is_fitted, validate_data
from sksurv.base import SurvivalAnalysisMixin
from sksurv.util import check_array_survival
from torch import nn

from torch_survival.progress import OptunaProgressCallback
from torch_survival.utils import merge_configs


class DeepHitNetwork(nn.Module):
    def __init__(
            self,
            n_inputs,
            n_times,
            hidden_layer_sizes=(32, 16),
            dropout=0.1,
            batch_norm=True,
    ):
        super().__init__()
        layers = []
        n_nodes = n_inputs
        for nodes in hidden_layer_sizes:
            layers.append(nn.Linear(n_nodes, nodes))
            if batch_norm:
                layers.append(nn.BatchNorm1d(nodes))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            n_nodes = nodes
        layers.append(nn.Linear(n_nodes, n_times))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DeepHitSearchSpace(TypedDict):
    #: Number of time points to use for discretization
    n_times: tuple[int, int] | int
    #: Weight for the ranking loss component
    alpha: tuple[float, float] | float
    #: Bandwidth of the radial basis function in the ranking loss component
    sigma: tuple[float, float] | float


class DeepHit(SurvivalAnalysisMixin, BaseEstimator):
    r""" Implements the DeepHit model presented by Lee et al. [1]_.

    Uses a deep neural network trained with the negative log likelihood of the survival time probability distribution,
    as estimated over discrete time intervals, combined with a ranking loss.

    .. [1] C. Lee, W. Zame, J. Yoon, and M. Van der Schaar, “DeepHit: A Deep Learning Approach to Survival Analysis
    With Competing Risks,” AAAI, vol. 32, no. 1, Apr. 2018, doi: 10.1609/aaai.v32i1.11842. Available:
    http://dx.doi.org/10.1609/aaai.v32i1.11842
    """

    #: Default hyperparameter search space
    default_search_space: DeepHitSearchSpace = {
        'n_times': (20, 100),
        'alpha': (0.0, 0.6),
        'sigma': (0.05, 1.0),
    }

    def __init__(
            self,
            search_space: DeepHitSearchSpace | None = None,
            hidden_layer_sizes=(32, 16),
            dropout=0.1,
            batch_norm=True,
            learning_rate=1e-3,
            n_epochs=100,
            batch_size=64,
            patience=10,
            n_trials=25,
            random_state=None,
            device=None,
    ):
        self.search_space = deepcopy(self.default_search_space)
        if search_space:
            self.search_space = merge_configs(self.search_space, search_space)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.n_trials = n_trials
        self.random_state = random_state
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _optimize(self, trial: optuna.Trial, X, y_event, y_time):
        # Do 5-fold cross validation
        scores = []
        for train_idx, test_idx in StratifiedKFold(n_splits=5).split(X, y_event):
            model, _ = self._train(trial, X[train_idx], y_event[train_idx], y_time[train_idx])
            surv = model.predict_surv_df(X[test_idx])
            c_index = EvalSurv(surv, y_time[test_idx], y_event[test_idx], censor_surv='km').concordance_td('antolini')
            scores.append(c_index)
        return - sum(scores) / len(scores)

    def _train(self, trial: optuna.Trial, X, y_event, y_time):
        n_inputs, n_outputs = X.shape[-1], 1

        # Discretize event times
        n_times = self.search_space['n_times']
        if not isinstance(n_times, int):
            n_times = trial.suggest_int('n_times', *n_times)
        y_trans = LabTransDiscreteTime(n_times)
        y = y_trans.fit_transform(y_time, y_event)

        # Split into training and validation for early stopping
        X_train, X_val, y_time_train, y_time_val, y_event_train, y_event_val = \
            train_test_split(X, *y, test_size=0.2, stratify=y_event, random_state=self.random_state)
        y_train = (y_time_train, y_event_train)
        y_val = (y_time_val, y_event_val)

        # Sample loss parameters
        alpha = self.search_space['alpha']
        if not isinstance(alpha, (int, float)):
            alpha = trial.suggest_float('alpha', *alpha)
        alpha = float(alpha)
        sigma = self.search_space['sigma']
        if not isinstance(sigma, (int, float)):
            sigma = trial.suggest_float('sigma', *sigma, log=True)
        sigma = float(sigma)

        # Train and return model
        net = DeepHitNetwork(
            n_inputs,
            y_trans.out_features,
            hidden_layer_sizes=self.hidden_layer_sizes,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
        )
        model = DeepHitSingle(net, tt.optim.Adam(lr=self.learning_rate), duration_index=y_trans.cuts,
                              alpha=alpha, sigma=sigma, device=self.device)
        callbacks = [tt.callbacks.EarlyStopping(patience=self.patience)]
        model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.n_epochs, callbacks=callbacks,
                  val_data=(X_val, y_val), verbose=False)
        return model, y_trans.cuts

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
        X = X.astype(np.float32)
        y_event, y_time = check_array_survival(X, y)

        # Seed PyTorch random number generator
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        # Optimize hyperparameters
        optuna.logging.disable_default_handler()
        warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
        with OptunaProgressCallback(model_name='DeepHit', n_trials=self.n_trials) as callback:
            study = optuna.create_study(sampler=TPESampler(seed=self.random_state) if self.random_state else None)
            objective = functools.partial(self._optimize, X=X, y_event=y_event, y_time=y_time)
            study.optimize(objective, n_trials=self.n_trials, callbacks=[callback])

        # Train model
        self.optuna_params_ = study.best_params
        self.model_, self.disc_times_ = self._train(study.best_trial, X, y_event, y_time)

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
        X = X.astype(np.float32)
        pmf = self.model_.predict_pmf(X)
        times = pmf @ self.disc_times_
        return times

    def get_optuna_params(self):
        return self.optuna_params_
