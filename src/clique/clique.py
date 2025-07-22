from re import search
from pathlib import Path
from typing_extensions import Self, Any, Iterable, Protocol, runtime_checkable
from sklearn.metrics import mean_absolute_error
from sklearn.utils.validation import check_is_fitted
from numpy import ndarray, nan, zeros, nan_to_num, isnan
from numpy.ma import masked_invalid
from pandas import DataFrame, Series
from joblib import dump, load
from tensorflow import keras
from keras.models import load_model


@runtime_checkable
class IModel(Protocol):
    '''
    Interface wrapper for multi-model support (namely, `fit` and `predict`).
    For maximum coverage, ignores functions like `get_params` which are included in scikit-learn's `BaseEstimator`.
    '''
    def fit(self, X, y, **kwargs) -> Self: ...
    def predict(self, X, **kwargs) -> ndarray: ...


class ModelProfile:
    '''
    Wrapper class to store models with `fit` and `predict` keyword arguments, plus error scores post-evaluation.
    Calls to `fit` and `predict` are wrapped to pass stored keywords, although `kwargs` can still be added on the fly.
    Beyond that, calls to the profile will generally behave as calls to the underlying model class.
    '''
    
    def __init__(self, model:IModel, fit_kw:dict={}, predict_kw:dict={}) -> None:
        '''
        Creates an instance of the class. 
        `model` must be compatible with the `IModel` interface, 
            and `fit_kw` and `predict_kw` must be of type `dict` if provided.    
        '''
        if not isinstance(model, IModel): raise ValueError('`model` must be compatible with the IModel interface.')
        if not isinstance(fit_kw, dict): raise ValueError('`fit_kw` must be of type `dict`.')
        if not isinstance(predict_kw, dict): raise ValueError('`predict_kw` must be of type `dict`.')
        self._model = model
        self.fit_kw = fit_kw
        self.predict_kw = predict_kw
        self.model_name = model.name if hasattr(model, 'name') else self.model_type
        self.score = nan # initialized to ensure a (non-)value is always available

    # object overrides

    def __getattr__(self, name) -> Any:
        '''Override to pull in attributes from the underlying model.'''
        return getattr(self._model, name)
    
    def __repr__(self) -> str:
        '''Override to indicate a ModelProfile (vs an IModel) when prompted.'''
        return f'<ModelProfile ({self.model_type})>'
    
    # properties

    @property
    def model_type(self) -> str:
        '''Returns the type of the underlying model.'''
        return f'{type(self._model).__name__}'
    
    # IModel wrappers

    def fit(self, X, y, **kwargs) -> IModel:
        '''Wrapper for `fit` to automatically pass stored `fit` keywords.'''
        return self._model.fit(X, y, **kwargs, **self.fit_kw)
    
    def predict(self, X, **kwargs) -> ndarray:
        '''Wrapper for `predict` to automatically pass stored `predict` keywords.'''
        return self._model.predict(X, **kwargs, **self.predict_kw)


class EvaluationError(Exception):
    '''Specific for errors during Clique evaluation.'''
    pass


class Clique(dict):
    '''
    An ensemble of models that, when provided with test data, validates new additions to prevent downgrades in performance.
    Instances may be initialized without an models or test data, but these must be present for the ensemble to be selective.
    Inherits from `dict` and supports the `IModel` protocol.
    '''
    def __init__(self, **kwargs) -> None:
        '''
        Creates a new instance of the class. Accepts the following keyword arguments to enable model evaluations:
            - `models`: The initial model or models to add. Not required, but at least 1 must be present to make predictions.
            - `inputs`: A set of inputs for model scoring. Must be set to evaluate and reject new models.
            - `targets`: A set of targets for model scoring. Must be set to evaluate and reject new models.
            - `scoring`: The scoring function to use for model evaluation. Defauls to mean absolute error if none is provided.
            - `limit`: The target size of the ensemble. Defaults to the number of models provided at construction.
        '''
        self._is_fitted = False
        self.limit = kwargs.get('limit') or len(self) # must precede model setup or it will fail
        self.scoring = kwargs.get('scoring') or mean_absolute_error # above may apply to this too
        self.reset_test_data(inputs=kwargs.get('inputs'), targets=kwargs.get('targets'))
        model_or_models = kwargs.get('models')
        match model_or_models:
            case IModel():
                model = ModelProfile(model_or_models)
                self[model.model_name] = model
            case ModelProfile():
                self[model_or_models.model_name] = model_or_models
            case list() | dict():
                if isinstance(model_or_models, dict): model_or_models = model_or_models.values()
                for t in model_or_models:
                    model = t if isinstance(t, ModelProfile) else ModelProfile(t) # ModelProfile constructor will type check
                    self[model.model_name] = model
            case str() | Path():
                if isinstance(model_or_models, str): model_or_models = Path(model_or_models)
                self.load(load_dir=model_or_models)
            case None: pass
            case _: raise ValueError('`models` must be an `IModel`, `ModelProfile`, or a list of such objects.')
        # self.inputs = kwargs.get('inputs')
        # self.targets = kwargs.get('targets')

    # dict overrides

    def __setitem__(self, key, value) -> None:
        '''Override to do type checking, avoid naming collisions, and evaluate new models as they are added.'''
        if key in self: # prevent naming overlap -- recurses if key already exists
            match_obj = search(r'_\d+$', key)
            if match_obj:key = f'{key[:match_obj.start()]}_{int(key[match_obj.start()+1:])+1}'
            else: key += '_0'
            value.model_name = key
            self.__setitem__(key, value)
        if not isinstance(value, ModelProfile): raise ValueError('Clique can only contain items of type ModelProfile.')
        if len(self) > self.limit and self.can_evaluate and self.scoring(self.targets, value.predict(self.inputs)) > self.mean_score: return None
        return super().__setitem__(key, value)
    
    def __setattr__(self, name, value):
        '''
        Override to add type checking for updates to `inputs`, `targets`, `scoring`, and `limit`.
        `limit` must be a positive integer. A value of 0 is treated as no limit.
        `scoring` must be a function that accepts (<true targets>, <predicted targets>) as input, although only callability is checked.
        `inputs` and `targets` must be of type `ndarray`, `DataFrame`, `Series`, or `None`, and match the length of the other
            field if that has already been set. (To apply a new set of testing data, use `reset_test_data` instead).
        '''
        match name:
            case 'limit': 
                if int(value) < 0: raise ValueError('Limit must be a positive integer. Set to 0 for no limit.')
            case 'scoring':
                if not callable(value): raise ValueError('Scoring function must be a callable function.')
            case 'inputs' | 'targets':
                match value:
                    case ndarray() | DataFrame() | Series() | None: pass
                    case _: raise ValueError('Testing data must be of type `ndarray`, `DataFrame`, `Series`, or `None`.')
                if value is not None and self.testing_initialized:
                    _compare = self.targets if name == 'inputs' else self.inputs
                    if _compare is not None:
                        len_value = value.shape[0] if isinstance(value, ndarray) else len(value)
                        len_compare = _compare.shape[0] if isinstance(_compare, ndarray) else len(_compare)
                        if len_value != len_compare: raise ValueError('Data length for testing inputs and targets must match.')
        object.__setattr__(self, name, value)

    def __iter__(self) -> Iterable:
        '''Override to yield values on for loops by default.'''
        return iter(self.values())

    def __repr__(self) -> str:
        return f'<Clique ({len(self)} model(s); limit: {self.limit if self.limit > 0 else "none"})>'
    
    # properties

    @property
    def mean_score(self) -> float:
        '''Returns the mean error score for all models in the collection.'''
        return sum([model.score for model in self]) / len(self) if len(self) > 0 else nan
    
    @property
    def best_score(self) -> float:
        '''Returns the lowest error score of all models in the collection.'''
        return min([model.score for model in self]) if len(self) > 0 else nan
    
    @property
    def best_model(self) -> IModel:
        '''Returns the model in the collection with the best (lowest) error score.'''
        best_score = self.best_score
        for model in self:
            if model.score == best_score: return model

    @property
    def testing_initialized(self) -> bool:
        '''Returns whether test inputs and targets have been initialized. Returns `True` even if either is set to `None`.'''
        return hasattr(self, 'inputs') and hasattr(self, 'targets')

    @property
    def can_evaluate(self) -> bool:
        '''Returns whether the ensemble is ready for evaluation (i.e., models have been fit and testing data is present).'''
        return self.testing_initialized and self.inputs is not None and self.targets is not None and self._is_fitted
    
    # object management

    def copy(self) -> Self:
        '''
        Override so copies of this instance have the same test data and scoring function.
        Returns a copy of the instance.
        '''
        clone = Clique(models=super().copy())
        clone._is_fitted = self._is_fitted
        clone.reset_test_data(inputs=self.inputs, targets=self.targets)
        clone.scoring = self.scoring
        clone.limit = self.limit
        return clone
    
    def save(self, save_dir:str|Path) -> Self:
        '''
        Saves the current ensemble of models to disk under `save_dir`. 
        If `save_dir` does not exist, it (and any parent folders) will be created before saving.
        Returns the instance for method chaining.
        '''
        if isinstance(save_dir, str): save_dir = Path(save_dir)
        if len(save_dir.suffix) > 0: raise ValueError('`save_dir` must be a directory path.')
        save_dir.mkdir(parents=True, exist_ok=True)
        for model_id, model in self.items():
            if model.model_type in ['Sequential', 'Model']: model.save(save_dir.joinpath(f'{model_id}.keras'))
            else: dump(model._model, save_dir.joinpath(f'{model_id}.joblib')) # remember -- save the underlying model and NOT the profile!
        return self
    
    def load(self, load_dir:str|Path) -> Self:
        '''
        Loads all models saved in `load_dir` into the collection, then preps the ensemble for evaluation.
        If `limit` is set and the ensemble is ready for evaluation, loaded models will be evaluated and (possibly) rejected.
        Returns the instance for method chaining.
        '''
        if len(self) == 0 and not self._is_fitted: self._is_fitted = True # hack to make loading pre-trained models into an empty Clique work
        if isinstance(load_dir, str): load_dir = Path(load_dir)
        if not load_dir.is_dir(): raise ValueError('`load_dir` must be an existing directory.')
        for filepath in load_dir.iterdir():
            match filepath.suffix:
                case '.keras': 
                    model = load_model(filepath=filepath)
                    if not hasattr(model, 'optimizer'): self._is_fitted = False
                case '.joblib': 
                    model = load(filename=filepath)
                    try: check_is_fitted(model)
                    except: 
                        if hasattr(model, '_random_seed'): continue # Catboost https://github.com/catboost/catboost/blob/master/catboost/python-package/catboost/core.py#L1769
                        self._is_fitted = False
                case _: continue
            model = ModelProfile(model)
            self[model.model_name] = model
        return self
    
    # training/deployment

    def reset_test_data(self, inputs:ndarray|DataFrame|Series|None=None, targets:ndarray|DataFrame|Series|None=None) -> Self:
        '''
        Allows both `inputs` and `targets` to be swapped out with a single function call.
        (As compared to clearing one, then reassigning the other - or both - attributes).
        Note that calling this function will clear all prior testing data, so only use when both sets should be overwritten.
        Returns the instance for method chaining.
        '''
        self.inputs = None
        self.targets = None
        self.inputs = inputs
        self.targets = targets
        return self

    def fit(self, X:ndarray|DataFrame|Series, y:ndarray|DataFrame|Series, **kwargs) -> Self:
        '''
        Fits each of the models in the ensemble to a set of training data, then evaluates them if training data is set.
        Supports `kwargs` as part of the `IModel` protocol, but is only recommended if all models are of the same type.
        Returns the instance for method chaining.
        '''
        errors = 0
        for model in self: 
            try: model.fit(X, y, **kwargs)
            except Exception as e: 
                errors += 1
                print(f'Exception occurred while training sub-model: {e}')
        self._is_fitted = self._is_fitted or (errors == 0)
        if self.can_evaluate: self.evaluate()
        return self
    
    def evaluate(self) -> Self:
        '''
        Scores all of the models in the ensemble against the ensemble's testing data and scoring function.
        Raises an error if the test data has not been set, or the scoring function is not configured properly.
        Returns the instance for method chaining.
        '''
        if self.inputs is None or self.targets is None: raise EvaluationError('Testing inputs and targets have not been defined.')
        if not self.can_evaluate: raise EvaluationError('Cannot evaluate models before they are trained. Call `fit` first.')
        for model in self: model.score = self.scoring(self.targets, model.predict(self.inputs))
        return self

    def prune(self, limit:int|None=None) -> Self:
        '''
        Removes models with error scores above the mean. If `limit` > 0, recurses until `len()` <= `limit`.
        Returns a copy of the instance with models at or below `limit`.
        '''
        mean_score = self.mean_score # instantiated since property loops through collection every time
        if isnan(mean_score): raise EvaluationError('Model has not been evaluated yet. Call `evaluate` first.')
        clone = self.copy()
        clone.limit = limit or self.limit
        for model_id in self.keys():
            if isnan(self[model_id].score): raise EvaluationError('Some models have not been scored. Call `evaluate` first.')
            if self[model_id].score > mean_score: del clone[model_id]
        if len(clone) > clone.limit > 1: return clone.prune(limit=limit)
        return clone

    def predict(self, X:ndarray|DataFrame|Series, **kwargs) -> ndarray:
        '''
        Makes predictions for all models in the ensemble, then returns the consensus (mean) for all results.
        Supports `kwargs` as part of the `IModel` protocol, but is only recommended if all models are of the same type.
        Returns the instance for method chaining.
        '''
        consensus = zeros(len(X))
        for model in self:
            predictions = model.predict(X, **kwargs)
            predictions = predictions.reshape(-1) # reshape needed for tensorflow output; doesn't impact other model types
            mask = masked_invalid(predictions) # mask NaN and +/- inf to find greatest legit values; https://stackoverflow.com/a/41097911/3178898
            predictions = nan_to_num(predictions, posinf=mask.max()+mask.std(), neginf=mask.min()-mask.std()) # then use those to clamp the invalid ones
            consensus += predictions
        consensus = consensus / len(self)
        return consensus