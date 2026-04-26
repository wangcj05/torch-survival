<div align="center">
  <img width="150rem" alt="torch-survival Logo" src="https://github.com/taltstidl/torch-survival/blob/main/assets/logo.svg" />
  <h1 align="center">torch-survival: A PyTorch-based library for Deep Survival Analysis</h1>

  [![docs](https://github.com/taltstidl/torch-survival/actions/workflows/docs.yaml/badge.svg)](https://github.com/taltstidl/torch-survival/actions/workflows/docs.yaml) 
</div>

**`torch-survival`** is a library built upon [PyTorch](https://pytorch.org) and [scikit-survival](https://github.com/sebp/scikit-survival) to make survival analysis using deep learning more accessible.

It's main goal is to implement a diverse set of deep learning methods for survival analysis using `scikit-survival`'s API design, thus offering an improved out-of-box experience compared to existing libraries (see [alternatives](#alternatives)). In addition, it provides many common loss functions and metrics that may also be used standalone in other PyTorch-based projects.

> [!IMPORTANT]
> While this library has been extensively tested and evaluated as part of the [SurvHub](https://github.com/taltstidl/survhub) benchmark, some parts of the API design and documentation are still considered a work in progress. Expect breaking changes as we converge towards a consistent and enjoyable developer experience.

## Installation

**`torch-survival`** is available from PyPI and only depends on PyTorch, `scikit-survival`, and (for now) `pycox`. To install:

```bash
pip install torch-survival
```

*For now only alpha releases are available.*

## Getting Started

Since the API design of **`torch-survival`** closely mimics that of `scikit-survival` (and in extension `scikit-learn`) it only takes a few lines of code to get started:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.datasets import load_whas500
from torch_survival.models import DeepSurv

X, y = load_whas500()
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = DeepSurv(random_state=42, device='cpu')
model.fit(X_train, y_train)
c_index = model.score(X_test, y_test)
```

## Implemented Models

This list summarizes the currently available models. We aim to steadily improve coverage and also welcome community contributions.

* **DeepSurv** from Katzman *et al.*: [DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network](https://link.springer.com/article/10.1186/s12874-018-0482-1) (BMC Medical Research Methodology 2018)
* **DeepHit** from Lee *et al.*: [DeepHit: A Deep Learning Approach to Survival Analysis With Competing Risks](https://ojs.aaai.org/index.php/AAAI/article/view/11842) (AAAI 2018)
* **DeepWeiSurv** from Bennis *et al.*: [Estimation of Conditional Mixture Weibull Distribution with Right Censored Data Using Neural Network for Time-to-Event Analysis](https://link.springer.com/chapter/10.1007/978-3-030-47426-3_53) (PAKDD 2020)
* **RankDeepSurv** from Jing *et al.*: [A deep survival analysis method based on ranking](https://www.sciencedirect.com/science/article/pii/S0933365718305992) (Artificial Intelligence in Medicine 2019)

## Alternatives

While **torch-survival** is the first comprehensive and production-ready library for deep survival analysis, there are other related libraries that you may want to consider:

* [**`pycox`**](https://github.com/havakv/pycox) has building blocks for both continuous-time models (think DeepSurv) and discrete-time models (think DeepHit) but lacks ready-to-use models.
* [**`torchsurv`**](https://github.com/Novartis/torchsurv) implements loss functions and metrics with a heavier focus on statistical evaluation at the expense of computational performance.
