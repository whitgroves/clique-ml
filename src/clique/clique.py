from typing_extensions import Self, Any, Callable, Protocol, runtime_checkable
from numpy import ndarray, nan
from pandas import DataFrame, Series

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
        if not isinstance(model, IModel): raise ValueError('`model` must be compatible with the IModel interface.')
        if not isinstance(fit_kw, dict): raise ValueError('`fit_kw` must be of type `dict`.')
        if not isinstance(predict_kw, dict): raise ValueError('`predict_kw` must be of type `dict`.')
        self._model = model
        self._fit_kw = fit_kw
        self._predict_kw = predict_kw
        self.model_type = f'{type(model).__name__}'
        self.score = nan # initialized to ensure a (non-)value is always available

    def __getattr__(self, name) -> Any:
        return getattr(self._model, name)
    
    def __repr__(self) -> str:
        return f'<ModelProfile ({self.model_type})>'
    
    def fit(self, X, y, **kwargs) -> IModel:
        return self._model.fit(X, y, **kwargs, **self._fit_kw)
    
    def predict(self, X, **kwargs) -> ndarray:
        return self._model.predict(X, **kwargs, **self._predict_kw)


from sklearn.metrics import mean_absolute_error
from re import search

class Clique(dict):
    '''
    An ensemble of models that, when provided with test data, validates new additions to prevent downgrades in performance.
    Instances may be initialized without an models or test data, but these must be present for the ensemble to be selective.
    Inherits from `dict`
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
        model_or_models = kwargs.get('models')
        match model_or_models:
            case IModel():
                model = ModelProfile(model_or_models)
                self[model.model_type] = model
            case ModelProfile():
                self[model_or_models.model_type] = model_or_models
            case list():
                for t in model_or_models:
                    model = t if isinstance(t, ModelProfile) else ModelProfile(t) # ModelProfile constructor will type check
                    self[model.model_type] = model
            case None: pass
            case _: raise ValueError('`models` must be an `IModel`, `ModelProfile`, or a list of such objects.')
        self.inputs = kwargs.get('inputs')
        self.targets = kwargs.get('targets')
        self.scoring = kwargs.get('scoring') or mean_absolute_error
        self.limit = kwargs.get('limit') or len(self) or nan

    def __setitem__(self, key, value) -> None:
        '''Override to do type checking and avoid naming collisions as new models are added.'''
        if key in self: # prevent naming overlap -- recurses if key already exists
            match_obj = search(r'_\d+$', key)
            if match_obj:key = f'{key[:match_obj.start()]}_{int(key[match_obj.start()+1:])+1}'
            else: key += '_0'
            self.__setitem__(key, value)
        if not isinstance(value, ModelProfile): raise ValueError('Clique can only contain items of type ModelProfile.')
        # TODO: scoring, evaluation
        return super().__setitem__(key, value)
    
    def __setattr__(self, name, value):
        '''Override to add type checking for updates to `inputs`, `targets`, `scoring`, and `limit`.'''
        match name:
            case 'limit': value = float(value)
            case 'scoring':
                if not callable(value): raise ValueError('Scoring function must be a callable function.')
            case 'inputs' | 'targets':
                match value:
                    case ndarray() | DataFrame() | Series() | None: pass
                    case _: raise ValueError('Testing data must be of type `ndarray`, `DataFrame`, `Series`, or `None`.')
                if value is not None:
                    _compare = self.targets if name == 'inputs' else self.inputs
                    if _compare is not None:
                        len_value = value.shape[0] if isinstance(value, ndarray) else len(value)
                        len_compare = _compare.shape[0] if isinstance(_compare, ndarray) else len(_compare)
                        if len_value != len_compare: raise ValueError('Data length for testing inputs and targets must match.')
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        return f'<Clique ({len(self)} model(s); limit: {self.limit})>'
    
    def reset_testing_data(self, inputs:ndarray|DataFrame|Series|None=None, targets:ndarray|DataFrame|Series|None=None) -> None:
        '''
        Allows both `inputs` and `targets` to be swapped out with a single function call.
        (As compared to clearing one, then reassigning the other - or both - attributes).
        Note that calling this function will clear all prior testing data, so only use when both sets should be overwritten.
        '''
        self.inputs = None
        self.targets = None
        self.inputs = inputs
        self.targets = targets
    
    def evaluate_models(self) -> float:
        '''
        Scores all of the models in the ensemble against the ensemble's testing data and scoring function.
        Raises an error if the test data has not been set, or the scoring function is not configured properly.
        Returns the mean score for the ensemble, which is also stored in the class.
        '''
        # for model in self._models
        pass
            
    # def train(self) -> None:
    #     for model in self._models:
    #         model.fit(...)