import functools
import warnings

import optuna
import torch
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted, validate_data
from sksurv.base import SurvivalAnalysisMixin
from sksurv.util import check_array_survival

from torch_survival.losses import cox_neg_log_likelihood
from torch_survival.metrics import concordance_index
from torch_survival.progress import OptunaProgressCallback
from torch_survival.sample import OptunaSampler, TunedTopology, TunedCategorical, TunedFloat


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

    Parameters
    ----------
    hidden_layer_sizes: list of ints or TunedTopology, default=TunedTopology(n_layers=4, n_neurons=50)
        If a list, the ith element represents the number of neurons in the ith hidden layer. Alternatively, a tunable
        topology specifies the maximum number of layers and of neurons per layer.
    activation: {'relu', 'selu'} or TunedCategorical, default=TunedCategorical(choices=['relu', 'selu'])
        Activation function for the hidden layer.
    dropout: float or TunedFloat, default=TunedFloat(low=0.0, high=0.5)
        Dropout probability for the hidden layer.
    optimizer: {'sgd', 'adam'} or TunedCategorical, default=TunedCategorical(choices=['sgd', 'adam'])
        Optimizer used for weight optimization.
    learning_rate: float or TunedFloat, default=TunedFloat(low=1e-7, high=1e-3, log=True)
        Initial learning rate used when optimizing weights.
    momentum: float or TunedFloat, default=TunedFloat(low=0.8, high=0.95)
        Momentum or first moment vector
    scheduler: {'inverse_time'} or TunedCategorical, default='inverse_time'
        Scheduler used for weight updates.
    decay: float or TunedFloat(low=0.0, high=0.001)
        Decay used by scheduler when updating learning rate.
    n_epochs: int, default=500
        Number of training epochs (how many times each data point will be used).
    n_trials: int, default=50
        Number of hyperparameter optimization trials. Only relevant if tunable parameters are passed.
    random_state: int, default=None
        Determines random number generation for hyperparameter optimization and weight initialization. Pass an int for
        reproducible results across multiple function calls.
    device: str or torch.device, default=None
        Device on which tensors will be allocated. If None, uses CUDA if available, else CPU.
    """

    def __init__(
            self,
            hidden_layer_sizes=TunedTopology(n_layers=4, n_neurons=50),
            activation=TunedCategorical(choices=['relu', 'selu']),
            dropout=TunedFloat(low=0.0, high=0.5),
            optimizer=TunedCategorical(choices=['sgd', 'adam']),
            learning_rate=TunedFloat(low=1e-7, high=1e-3, log=True),
            momentum=TunedFloat(low=0.8, high=0.95),
            scheduler='inverse_time',
            decay=TunedFloat(low=0.0, high=0.001),
            n_epochs=500,
            n_trials=50,
            random_state=None,
            device=None,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.dropout = dropout
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.scheduler = scheduler
        self.decay = decay
        self.n_epochs = n_epochs
        self.n_trials = n_trials
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

    def _train(self, trial: optuna.Trial | None, X, y_event, y_time):
        sampler = OptunaSampler(trial)

        # Set up model, optimizer, and scheduler
        n_inputs, n_outputs = X.shape[-1], 1
        model = sampler.sample_network(n_inputs, n_outputs, self.hidden_layer_sizes, self.activation, self.dropout)
        optimizer, scheduler = sampler.sample_optimizer(model, self.optimizer, self.learning_rate, self.momentum,
                                                        self.scheduler, self.decay)

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
        self.model_ = self._train(study.best_trial, X, y_event, y_time)
        self.model_.eval()

        # Override internal parameters with tuned hyperparameters
        best_params = study.best_params
        best_params['hidden_layer_sizes'] = [best_params.pop('n_neurons_' + str(i + 1)) for i in
                                             range(best_params.pop('n_layers'))]
        self.set_params(**best_params)

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

    def save(self, path):
        """ Save a trained model to disk.

        This saves the model's parameters (dictionary) and weights (tensors) such that the estimator can be fully
        restored using :meth:`DeepSurv.load`. Internally relies on :meth:`torch.save`.

        Parameters
        ----------
        path: str or file-like object or os.PathLike
            Path at which to save the model.
        """
        check_is_fitted(self)
        state = {
            'params': self.get_params(),
            'model': self.model_.state_dict(),
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path, device=None):
        """ Reload an already trained model.

        Parameters
        ----------
        path: str or file-like object or os.PathLike
            Path from which to load the model. See also :meth:`DeepSurv.save` for additional details.
        device: str or torch.device, default=None
            Device on which tensors will be allocated. If None, uses CUDA if available, else CPU.

        Returns
        -------
        model: DeepSurv
            The trained estimator.
        """
        # Note: Device handled manually as users may wish to use a device different from the original when restoring
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(path, map_location=device, weights_only=True)
        # Reload estimator with initial parameters
        params = state['params']
        params['device'] = device
        estimator = cls(**params)
        # Restore internal model based on weights
        sampler = OptunaSampler(None)
        n_inputs, n_outputs = next(iter(state['model'].values())).shape[-1], 1
        estimator.model_ = sampler.sample_network(n_inputs, n_outputs, estimator.hidden_layer_sizes,
                                                  estimator.activation, estimator.dropout)
        estimator.model_.load_state_dict(state['model'])
        estimator.model_.to(device)
        return estimator
