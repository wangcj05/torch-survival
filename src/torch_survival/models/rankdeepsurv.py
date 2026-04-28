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

from torch_survival.metrics import concordance_index
from torch_survival.progress import OptunaProgressCallback
from torch_survival.sample import OptunaSampler, TunedTopology, TunedCategorical, TunedFloat


class RankDeepSurv(SurvivalAnalysisMixin, BaseEstimator):
    r""" Implements the RankDeepSurv model presented by Jing et al. [1]_.

    Uses a deep neural network trained with a mean squared error and ranking loss, adapted for censoring, to estimate
    the survival time of each individual. This implementation tries to stay faithful to the original paper, with the
    following deviations:

    * Neither the original paper nor the provided implementation state how hyperparameters are derived. We use Optuna's
      default TPE sampler with 5-fold internal cross-validation here.
    * The original loss formulation uses separate :math:`\alpha` and :math:`\beta` weights for the regression and
      ranking loss, respectively. We instead use only a single :math:`\alpha` such that :math:`\mathcal{L}=
      \alpha \mathcal{L}_{mse} + (1-\alpha) \mathcal{L}_{rank}`.
    * Our implementation does not support or tune :math:`\ell_2` regularization. We found this to be detrimental to
      performance and were unable to fully replicate the described weight regularization.

    .. note::
       As the model predicts a single expected survival time, it does not support predicting either a cumulative hazard
       function or a survival function. This is not an omission but a fundamental consequence of the model's design.

    .. [1] B. Jing et al., “A deep survival analysis method based on ranking,” Artificial Intelligence in Medicine, vol.
       98, pp. 1–9, July 2019, doi: 10.1016/j.artmed.2019.06.001. Available:
       http://dx.doi.org/10.1016/j.artmed.2019.06.001

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
    decay: float or TunedFloat, default=TunedFloat(low=0.0, high=0.001)
        Decay used by scheduler when updating learning rate.
    alpha: float or TunedFloat, default=TunedFloat(low=0.0, high=1.0)
        Weight term for regression and ranking loss, with :math:`\mathcal{L}=\alpha \mathcal{L}_{mse} + (1-\alpha)
        \mathcal{L}_{rank}`.
    batch_size: int or TunedInt, default=128
        Size of minibatches, here the number of pairs as training iterates over pairs of compatible samples.
    n_epochs: int, default=500
        Number of training epochs (how many times each pair of compatible samples will be used).
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
            learning_rate=TunedFloat(low=1e-6, high=1e-2, log=True),
            momentum=TunedFloat(low=0.8, high=0.95),
            scheduler='inverse_time',
            decay=TunedFloat(low=0.0, high=0.001),
            alpha=TunedFloat(low=0.0, high=1.0),
            batch_size=128,
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
        self.alpha = alpha
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_trials = n_trials
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
        sampler = OptunaSampler(trial)

        # Set up model, optimizer, and scheduler
        n_inputs, n_outputs = X.shape[-1], 1
        model = sampler.sample_network(n_inputs, n_outputs, self.hidden_layer_sizes, self.activation, self.dropout)
        optimizer, scheduler = sampler.sample_optimizer(model, self.optimizer, self.learning_rate, self.momentum,
                                                        self.scheduler, self.decay)

        # Extract data pair indices for ranking loss
        event_i, event_j = y_event.unsqueeze(-2), y_event.unsqueeze(-1)
        time_i, time_j = y_time.unsqueeze(-2), y_time.unsqueeze(-1)
        # The compatibility matrix specifies which pairs (i,j) can be compared accounting for censoring
        comp = event_i & (time_i < time_j)
        pairs = torch.nonzero(comp)  # of shape (n_pairs, 2)

        # Sample batch size and loss parameters
        batch_size = sampler.sample_int('batch_size', self.batch_size)
        alpha = sampler.sample_float('alpha', self.alpha)

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
        self: RankDeepSurv
            The trained estimator.
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
