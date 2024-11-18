# clique-ml
A selective ensemble for predictive time-series models that tests new additions to prevent downgrades in performance.

This code was written and tested against a CUDA 12.2 environment; if you run into compatability issues, try setting up a `venv` using [`cuda-venv.sh`](./cuda-venv.sh).

## Usage
### Setup
```
pip install -U clique-ml
```
```
import clique
```
### Training

Create a list of models to train. Supports any class that can call `fit()`, `predict()`, and `get_params()`:
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
Data is automatically split for training, testing, and validaiton, so simply pass `models`, inputs (`X`) and targets(`y`) to `train_ensemble()`:

```
X, y = ... # preprocessed data; 20% is set aside for validation, and the rest is trained on using k-folds

ensemble = clique.train_ensemble(models, X, y, folds=5, limit=3) # instance of clique.SelectiveEnsemble
```
`folds` sets `n_splits` for [scikit-learn's `TimeSeriesSplit` class](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), which is used to implement k-folds here. For a single split, pass `folds=1`.

`limit` sets a **soft target** for how many models to include in the ensemble. When set, once exceeded, the ensemble will reject new models that raise its mean score.

By default, the ensemble trains using **5 folds** and **no size limit**.

### Evaluation

`train_ensemble()` will output the results of each sub-model's training on every fold:
```
Pre-training setup...Complete (0.0s)
Model 1/5: Fold 1/5: Stopped: PredictionError: Model is guessing a constant value. -- 3      
Model 2/5: Fold 1/5: Stopped: PredictionError: Model is guessing a constant value. -- 3       
Model 3/5: Fold 1/5: Accepted with score: 0.03233311 (0.1s) (CatBoostRegressor_1731893049_0)          
Model 3/5: Fold 2/5: Accepted with score: 0.02314115 (0.0s) (CatBoostRegressor_1731893050_1)          
Model 3/5: Fold 3/5: Accepted with score: 0.01777214 (0.0s) (CatBoostRegressor_1731893050_2)
...      
Model 5/5: Fold 2/5: Rejected with score: 0.97019375 (0.3s)                            
Model 5/5: Fold 3/5: Rejected with score: 0.41385662 (1.4s)                         
Model 5/5: Fold 4/5: Rejected with score: 0.41153231 (0.8s)          
Model 5/5: Fold 5/5: Rejected with score: 0.40335007 (1.6s)
```
Once trained, details of the final ensemble can be reviewed with:
```
print(ensemble) # <SelectiveEnsemble (5 model(s); mean: 0.03389993; best: 0.03321487; limit: 3)>
```
Or:
```
print(len(ensemble)) # 5
print(ensemble.mean_score) # 0.033899934449981864
print(ensemble.best_score) # 0.033214874389494775
```
### Pruning
Since `SelectiveEnsemble` has to accept the first *N* models to establish a mean, frontloading with weaker models may cause oversaturation, even when `limit` is set.

To remedy this, call `SelectiveEnsemble.prune()`:
```
pruned = ensemble.prune()
```
Which will return a copy of the ensemble with all sub-models scoring above the mean removed. 

If a `limit` is passed in, the removal of all models above the mean will recurse until that limit is reached:
```
triumvate = ensemble.prune(3)
print(len(ensemble)) # 3 (or less)
```
This recursion is automatic for instances where `SelectiveEnsemble.limit` is set manually or by `train_ensemble()`.

### Deployment
To make predictions, simply call:
```
predictions = ensemble.predict(...) # with a new set of inputs
```
Which will use the mean score across all sub-models for each prediction.

If you wish to continue training on an existing ensemble, use:
```
existing = clique.load_ensemble(X_test=X, y_test=y) # test data must be passed in for new model evaluation
updated = clique.train_ensemble(models, X, y, ensemble=existing)
```
Note that if a limit is set on the existing model, that will be set and enforced on the updated one.