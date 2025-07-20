# clique-ml
An ensemble for machine learning models that, when provided with test data, validates new additions to prevent downgrades in performance.

The container class `Clique` is compatible with any model that has both `fit` and `predict` methiods, although it has only been tested against regression models included in `tensorflow.`, `lightgbm`, `xgboost`, and `catboost`.

This code was written against a CUDA 12.2 environment; if you run into compatability issues, [`cuda-venv.sh`](./cuda-venv.sh) can be used to setup a virtualenv on linux.

All code is unlicensed and freely available for anyone to use. If you run into issues, please contact me on X: [@whitgroves](https://x.com/whitgroves)

## Classes

`clique` makes 4 classes available to the developer:

- `IModel`: A `Protocol` that acts as an interface for any model matching `scikit-learn`'s estimator API (`fit` and `predict`).

- `ModelProfile`: A wrapper class to bundle any `IModel` with `fit` and `predict` keyword arguments, plus error scores post-evaluation.

- `EvaluationError`: An error class for exceptions specific to model evaluation.

- `Clique`: A container class that provides the main functionality of the package, detailed below. Also supports `IModel`.

## Usage

### Installation

The package can be installed with `pip`:

```
pip install -U clique-ml
```

### Setup

First, prepare your data:

```
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./training_data.csv')
... # preprocessing
y = df['target_variable']
X = df.drop(['target_variable'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=489)
```

Then, instantiate the models that will form the core of the ensemble:

```
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import tensorflow as tf

models = [
    xgb.XGBRegressor(...),
    lgb.LGBMRegressor(...),
    cat.CatBoostRegressor(...),
    tf.keras.Sequential(...),
]
```

And load those models into your ensemble:

```
from clique import Clique

ensemble = Clique(models=models)
```

Now you can run your training loop (in this example, for time series data):

```
from sklearn.model_selection import TimeSeriesSplit

for i, (i_train, i_valid) in enumerate(TimeSeriesSplit().split(X_train)):
    val_data = [(X_train.iloc[i_valid, :], y_train.iloc[i_valid])]
    for model in ensemble: # for ModelProfile in Clique
        fit_kw = dict()
        predict_kw = dict()
        match model.model_type:
            case 'Sequential':
                if i == 0: model.compile(optimizer='adam', loss='mae')
                keras_kw = dict(verbose=0, batch_size=256)
                fit_kw.update(keras_kw)
                predict_kw.update(keras_kw)
            case 'LGBMRegressor':
                fit_kw.update(dict(eval_set=val_data, eval_metric='l1'))
            case 'XGBRegressor' | 'CatBoostRegressor':
                fit_kw.update(dict(verbose=0, eval_set=val_data))
        model.fit_kw = fit_kw
        model.predict_kw = predict_kw
    ensemble.fit(X_train.iloc[i_train, :], y_train.iloc[i_train])
```

And start to make predictions with the ensemble:

```
predictions = ensemble.predict(X_test)
```

### Evaluation

Once `fit` has been called, `Clique` can be evaluated and pruned to improve performance. 

To start, load your test data into the ensemble:

```
ensemble.inputs = X_test
ensemble.targets = y_test
```

Note that once either `inputs` or `targets` is set, the ensemble will not accept assignments to the other attribute if the data length does not match (e.g., if `inputs` has 40 rows, setting `targets` with data for 39 rows will raise a `ValueError`).

Consequently, the safest way to swap out testing data is through `reset_test_data`:

```
ensemble.reset_test_data(inputs=X_test, targets=y_test)
```

Which will first clear existing test data before assigning the new values (or clearing them, if no parameters are passed).

Then, call `evaluate` to score your models:

```
ensemble.evaluate()
```

Which will populate the `mean_score`, `best_score`, and `best_model` properties for the ensemble:

```
ensemble.           # <Clique (5 model(s); limit: none)>
ensemble.mean_score # 0.31446850398821063
ensemble.best_score # 0.033214874389494775
ensemble.best_model # <ModelProfile (CatBoostRegressor)>
```

An enable the `prune` function to return a copy of the ensemble with all models scoring above the mean removed:

```
triad = ensemble.prune(3) # <Clique (3 model(s); limit: 3)>
triad.mean_score          # 0.03373027543351623
```

### Deployment

Once trained and evaluated, the ensemble's models can be saved for later:

```
ensemble.save('.models/')
```

Which will save copies of each underlying model to be used elsewhere, or reloaded into another ensemble:

```
Clique().load('.models/') # fresh ensemble with no duplications
ensemble.load('.models/') # will duplicate all models in the collection
triad.load('.models/')    # will duplicate any saved models still in the collection
```

Also note that similar to `fit`, calls to `load` will enable evaluation and pruning, assuming all loaded models were previously trained.
