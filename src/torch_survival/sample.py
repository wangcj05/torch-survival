from typing import TypedDict, NotRequired

import optuna
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from torch import nn

from torch_survival.utils import make_activation


class TunedInt(TypedDict):
    low: int
    high: int
    step: NotRequired[int]
    log: NotRequired[bool]


class TunedFloat(TypedDict):
    low: float
    high: float
    step: NotRequired[float]
    log: NotRequired[bool]


class TunedCategorical(TypedDict):
    choices: list[str]


class TunedTopology(TypedDict):
    # Maximum number of hidden layers
    n_layers: int
    # Maximum number of neurons per hidden layer
    n_neurons: int


class TunedNeuralNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs, layers, activation, dropout):
        super().__init__()
        hidden = []
        n_nodes = n_inputs
        for nodes in layers:
            hidden.append(nn.Linear(n_nodes, nodes))
            hidden.append(make_activation(activation))
            hidden.append(nn.Dropout(p=dropout))
            n_nodes = nodes
        self.hidden = nn.Sequential(*hidden)
        self.output = nn.Linear(n_nodes, n_outputs)

    def forward(self, x):
        x = self.hidden(x)
        return self.output(x)


class OptunaSampler:
    def __init__(self, trial: optuna.Trial | None):
        self.trial = trial

    def sample_int(self, name: str, value: int | TunedInt) -> int:
        if isinstance(value, int):
            return value
        else:
            default = {'step': 1, 'log': False}
            return self.trial.suggest_int(name, **(default | value))

    def sample_float(self, name: str, value: float | TunedFloat) -> float:
        if isinstance(value, float):
            return value
        else:
            default = {'step': None, 'log': False}
            return self.trial.suggest_float(name, **(default | value))

    def sample_categorical(self, name: str, value: str | TunedCategorical) -> str:
        if isinstance(value, str):
            return value
        else:
            return self.trial.suggest_categorical(name, **value)

    def sample_network(
            self,
            n_inputs: int,
            n_outputs: int,
            layers: list[int] | TunedTopology,
            activation: str | TunedCategorical,
            dropout: float | TunedFloat,
    ) -> TunedNeuralNetwork:
        if not isinstance(layers, list):
            n_layers = self.trial.suggest_int('n_layers', 0, layers['n_layers'])
            layers = [self.trial.suggest_int('n_neurons_' + str(i + 1), 1, layers['n_neurons']) for i in
                      range(n_layers)]
        activation = self.sample_categorical('activation', activation)
        dropout = self.sample_float('dropout', dropout)
        return TunedNeuralNetwork(n_inputs, n_outputs, layers, activation, dropout)

    def sample_optimizer(
            self,
            model: nn.Module,
            optimizer: str | TunedCategorical,
            learning_rate: float | TunedFloat,
            momentum: float | TunedFloat,
            scheduler: str | TunedCategorical,
            decay: str | TunedFloat,
    ) -> tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler | None]:
        # Initialize optimizer
        optimizer_name = self.sample_categorical('optimizer', optimizer)
        learning_rate = self.sample_float('learning_rate', learning_rate)
        momentum = self.sample_float('momentum', momentum)
        optimizer = None
        if optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(momentum, 0.999))
        if optimizer is None:
            raise ValueError(f'Optimizer with name `{optimizer_name}` is not supported')
        # Initialize scheduler
        scheduler_name = self.sample_categorical('scheduler', scheduler)
        decay = self.sample_float('decay', decay)
        scheduler = None
        if scheduler_name == 'inverse_time':
            scheduler = sched.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + epoch * decay))
        # Return both
        return optimizer, scheduler
