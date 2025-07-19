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
        Creates a new instance of the class. Accepts the following keyword arguments to enable full functionality:
            - models: The initial model or models to add. Not required, but at least 1 must be present to make predictions.
            - test_X: A set of inputs to use for model evaluation. Must be set to enable rejection of new models.
            - test_y: A set of targets to use for model evaluation. Must be set to enable rejection of new models.
            - scoring: The scoring function to use for model evaluation. Defauls to mean absolute error if none is provided.
            - limit: The target size of the ensemble. Defaults to the number of models provided at construction.
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
            case _:
                raise ValueError('`models` must be an `IModel`, `ModelProfile`, or a list of such objects.')
        self._test_X = kwargs.get('test_X')
        self._test_y = kwargs.get('test_y')
        self._scoring = kwargs.get('scoring') or mean_absolute_error
        self.limit = kwargs.get('limit') or len(self) or None
    
    def __setitem__(self, key, value) -> None:
        '''Override to do type checking and avoid naming collisions as new models are added.'''
        if not isinstance(value, ModelProfile): raise ValueError('Clique can only contain items of type ModelProfile.')
        if key in self: # prevent naming overlap -- recurses if key already exists
            match_obj = search(r'_\d+$', key)
            if match_obj:key = f'{key[:match_obj.start()]}_{int(key[match_obj.start()+1:])+1}'
            else: key += '_0'
            self.__setitem__(key, value)
        return super().__setitem__(key, value)

    def __repr__(self) -> str:
        return f'<Clique ({len(self)} model(s); limit: {self.limit})>'
    
    def score(self, test_X:ndarray|DataFrame, test_y:ndarray|DataFrame|Series, scoring:Callable=None) -> float:
        '''
        Scores all of the models in the ensemble against a set of test data using the ensemble's scoring function.
        If an alternate scoring function is provided (`scoring`) that will be used instead.
        Returns the mean score for the ensemble, which is also stored in the class.
        '''
        # for model in self._models
        pass
            
    # def train(self) -> None:
    #     for model in self._models:
    #         model.fit(...)