import os
import gc
import time
import joblib
# import pickle
import psutil
import warnings
warnings.simplefilter('ignore') # ignore FutureWarnings; must precede pandas import
import pandas as pd
import numpy as np
import sklearn.metrics as met
import sklearn.model_selection as sel
import typing_extensions as ext # used over typing to backport 3.11+ features, specifically Self
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore bugged CUDA errors; must precede tf import
import tensorflow as tf
tf.keras.utils.disable_interactive_logging() # ensemble will provide its own condensed version
print(('GPU available.' if len(tf.config.list_physical_devices('GPU')) > 0 else 'No GPU detected. Run "sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm" to reload the GPU kernel.'))

# training classes for ensemble
class PredictionError(Exception): pass # specific for training feedback

class IModel(ext.Protocol): # partial wrapper for sklearn API
    def fit(self, X, y, **kwargs) -> ext.Self: ...
    def predict(self, X, **kwargs) -> np.ndarray: ...
    def get_params(self, deep=True) -> dict[str, ext.Any]: ...

class SelectiveEnsemble: # once len(models) >= limit, reject new models with scores above the mean
    def __init__(self, X_test:pd.DataFrame, y_test:pd.DataFrame|pd.Series, limit:int=None) -> None:
        self.limit = limit
        self.models = dict[str, IModel]()
        self.scores = dict[str, float]()
        self.kwargs = dict[str, dict]()
        self.test_x = X_test # TODO: generalize
        self.test_y = y_test
        
    @property
    def mean_score(self) -> float:
        return sum(self.scores[m] for m in self.models) / len(self) if len(self) > 0 else None
    
    @property
    def best_score(self) -> float:
        return min(self.scores[m] for m in self.models) if len(self) > 0 else None
    
    @property
    def best_model(self) -> tuple[IModel, str, dict]:
        return [(self.models[m], m, self.kwargs[m].copy()) for m in self.models if self.scores[m] == self.best_score][0]
    
    def add(self, model:IModel, name:str, kwargs:dict) -> tuple[bool, float]: # raises PredictionError
        if name in self.models: name = f'{name}(1)'
        pred = model.predict(self.test_x, **kwargs)
        if len(np.unique(pred)) == 1: raise PredictionError('Model is guessing a constant value.')
        if np.isnan(pred).any(): raise PredictionError('Model is guessing NaN.')
        score = met.mean_absolute_error(self.test_y, pred)
        if self.limit and len(self) >= self.limit and self.mean_score < score: return False, score
        self.models[name] = model
        self.scores[name] = score
        self.kwargs[name] = kwargs
        return True, score

    def prune(self, limit:int=None) -> ext.Self: # removes models with scores above the mean; recurses if limit is set
        pruned = SelectiveEnsemble(limit=(limit or self.limit))
        pruned.models = {m:self.models[m] for m in self.models if self.scores[m] <= self.mean_score}
        pruned.scores = {m:self.scores[m] for m in pruned.models}
        pruned.kwargs = {m:self.kwargs[m] for m in pruned.models}
        if pruned.limit and len(pruned) > pruned.limit > 1: return pruned.prune()
        return pruned
    
    def clone(self, limit:int=None) -> ext.Self:
        clone = SelectiveEnsemble(limit=(limit or self.limit))
        clone.models = self.models.copy()
        clone.scores = self.scores.copy()
        clone.kwargs = self.kwargs.copy()
        return clone
    
    def predict(self, X:pd.DataFrame, **kwargs) -> np.ndarray: # wrapper for soft voting; kwargs for compat
        y = np.zeros(len(X))
        for m in self.models:
            pred = self.models[m].predict(X, **self.kwargs[m])
            pred = pred.reshape(-1) # reshape needed for tensorflow output; doesn't impact other model types
            temp = np.ma.masked_invalid(pred) # mask NaN and +/- inf to find largest legit values; https://stackoverflow.com/a/41097911/3178898
            pred = np.nan_to_num(pred, posinf=temp.max()+temp.std(), neginf=temp.min()-temp.std()) # then use those to clamp the invalid ones
            y += pred
        y = y / len(self)
        return y

    def __len__(self) -> int:
        return len(self.models)
    
    def __repr__(self) -> str:
        return f'<SelectiveEnsemble ({len(self)} model(s)>' #; mean: {self.mean_score:.8f}; best: {self.best_score:.8f}; limit: {self.limit})>'
    
# training functions for ensemble

MODEL_FOLDER = '.models/'
if not os.path.exists(MODEL_FOLDER): os.makedirs(MODEL_FOLDER)
process = psutil.Process() # defaults to current process

# customize fit() and predict() kwargs for each model's type and params
def build_model_kwargs(model:IModel, val_data:tuple[pd.DataFrame, pd.Series]=None) -> tuple[dict, dict, dict]:
    fit_kw = dict()
    predict_kw = dict()
    early_stop_kw = dict()
    model_class = type(model).__name__
    match model_class:
        case 'Sequential':
            # model.compile(optimizer='adam', loss='mae')
            keras_kw = dict(batch_size=256, verbose=0)
            fit_kw.update(keras_kw)
            predict_kw.update(keras_kw)
            early_stop_kw['validation_data'] = val_data
        case 'LGBMRegressor':
            fit_kw.update(dict(verbose=False)) # verbose=0 throws an error
            if 'early_stopping_round' in model.get_params():
                early_stop_kw['eval_set'] = [val_data]
                early_stop_kw['eval_metric'] = 'l1'
        case 'XGBRegressor' | 'CatBoostRegressor':
            fit_kw.update(dict(verbose=0))
            if 'early_stopping_rounds' in model.get_params():
                early_stop_kw['eval_set'] = [val_data]
    fit_kw.update(early_stop_kw)
    return fit_kw, predict_kw, early_stop_kw

# builds an ensemble trained on the data from load_vars(). if an existing ensemble is provided, it will be updated instead.
def train_ensemble(models:list[IModel], X:pd.DataFrame, y:pd.DataFrame|pd.Series, folds:int=5, limit:int=None, ensemble:SelectiveEnsemble=None) -> SelectiveEnsemble:
    setup_start = time.time()
    print(f'Pre-training setup...', end='\r')
    os.makedirs('.models/', exist_ok=True)
    cutoff = int(len(X)*0.8) # 80:20 train/test split
    X_train, X_test = X[:cutoff], X[cutoff:-1]
    y_train, y_test = y[:cutoff], y[cutoff:-1]
    ensemble = ensemble.clone(limit=(limit or len(ensemble))) if ensemble else SelectiveEnsemble(X_test, y_test, limit=(limit or len(models)))
    cv = sel.TimeSeriesSplit(folds)
    _X, _y = X_train, y_train
    setup_time = time.time() - setup_start
    print(f'Pre-training setup...Complete ({setup_time:.1f}s)')
    for j, model in enumerate(models): # each model gets its own ensemble, then the best fold will be added to the main
        name = type(model).__name__
        is_sequential = name == 'Sequential'
        if is_sequential:
            model.compile(optimizer='adam', loss='mae')
            name = model.name            
        _msg = f'Model {j+1}/{len(models)}:'
        for i, (i_train, i_valid) in enumerate(cv.split(_X)):
            progress = 0
            try: # fail gracefully instead of giving up on the whole ensemble
                fold_start = time.time()
                _name = f'{name}_{int(time.time())}_{i}'
                msg = f'{_msg} Fold {i+1}/{folds}:'
                print(f'{msg} Training {name}...'+' '*48, end='\r')
                progress = 1
                X_valid, y_valid = _X.iloc[i_valid, :], _y.iloc[i_valid]
                fit_kw, predict_kw, early_stop_kw = build_model_kwargs(model, (X_valid, y_valid))
                try: model.fit(_X.iloc[i_train, :], _y.iloc[i_train], **fit_kw)           # if fit kwargs fail...
                except: model.fit(_X.iloc[i_train, :], _y.iloc[i_train], **early_stop_kw) # fallback to early stop only
                del X_valid, y_valid
                progress = 2
                # mem_total = process.memory_info().rss / 1024**3 # B -> GiB
                # if mem_total > MEMORY_CAP: raise MemoryError(f'High memory allocation ({mem_total:.1f} > {MEMORY_CAP:.1f} GiB)') # plug memory leak
                print(f'{msg} Adding {name} to ensemble...', end='\r')
                if is_sequential: # TODO: remove
                    clone = tf.keras.models.clone_model(model)
                    clone.set_weights(model.get_weights())
                else: clone = None
                progress = 3
                res, score = ensemble.add((clone or model), _name, predict_kw)
                progress = 4
                if (res):
                    filepath = os.path.join(MODEL_FOLDER, _name)
                    if is_sequential: model.save(f'{filepath}.keras')
                    else: joblib.dump(model, f'{filepath}.joblib')
                fold_time = time.time()-fold_start
                print(f'{msg} {("Accepted" if res else "Rejected")} with score: {score:.8f}' +f' ({fold_time:.1f}s)'+(f' ({_name})' if res else '')+' '*10)
                    #  +f' ({fold_time:.1f}s) ({mem_total:.1f} GiB)'+(f' ({_name})' if res else '')+' '*10)
                progress = 5
            except Exception as e:
                print(f'{msg} Stopped: {type(e).__name__}: {e} -- {progress}')
                if isinstance(e, PredictionError): break # these tend not to improve, so move on to the next model
                if isinstance(e, MemoryError): break     # malloc resets with each model, so move on if exceeded
            finally:
                while gc.collect() > 0: pass # it had to be this way
    return ensemble

def load_ensemble(model_dir:str=MODEL_FOLDER, **kwargs) -> SelectiveEnsemble:
    ensemble = SelectiveEnsemble(**kwargs)
    for file in os.listdir(model_dir):
        name, filetype = file.rsplit('.', 1)
        filepath = os.path.join(model_dir, file)
        if filetype == 'keras': model = tf.keras.models.load_model(filepath)
        else: model = joblib.load(filepath)
        kwargs = build_model_kwargs(model, (ensemble.test_x, ensemble.test_y))[1] # only need predict_kw
        ensemble.add(model, name, kwargs)
    if len(ensemble) == 0: raise FileNotFoundError(f'No models saved in {model_dir}.')
    return ensemble