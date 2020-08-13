import numpy as np 
import itertools
import pickle

class PolynomialFeatures:
    """
    Transform Linear Features to Polynomial Features

    Parameters
    ----------
    degree : polynomial degree
    """
    def __init__(self, degree=3):
        self._degree = degree
        
    def fit_transform(self, X):
        """
        Transforms features

        Parameters:
        ----------
        X : ndarray of shape (observations, features)
            X values to be transformed

        Returns:
        -------
        phi : ndarray of shape (observations, (degree+1) * features)
            Transformed X values
        """
            
        n = X.shape[0]
        phi = X
        for i in range(1, self._degree + 1):
            phi = np.hstack([phi, X ** i])

        return phi
        

class CosineFeatures:    
    """
    TO BE TESTED !

    
    Parameters
    ----------
    D : degree
    T : period
    """
    def __init__(self, degree=1, period=2):
        self._degree = degree
        self._period = period
        
    def fit_transform(self, X):
        
        phi = np.ones((len(X), self._degree + 1)) 
        for d in range(1, self._degree + 1):
            phi[:, d] = np.cos(d * 2 * np.pi / self._degree / T * x)
        
        self._transformed = True
        return phi


class NonLinearRegression:
    """
    Non Linear Regression by using a Scikit-Learn like base model.

    Supports Polynomial or Cosine Features.

    Parameters:
    ----------
    model : Scikit-Learn like object

    kind : string 
        indicating whether to use 
        polynomial or cosine features

    transform_params : dictionary 
        Contains the parameters 
        needed to transform the features

    model_params : dictionary (optional)  
        Contains the parameters to be used with the model

    Example:
    -------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> # y = 1 * x_0 + 2 * x_1 + 3
    >>> y = np.dot(X, np.array([1, 2])) + 3
    >>> params = {'degree' : 2}
    >>> reg = NonLinearRegression(LinearRegression, kind='poly', transform_params=params).fit(X, y)
    >>> reg.model.score(X, y)
    xxxxxxx
    >>> reg.model.coef_
    array([xx, xx])
    >>> reg.model.intercept_
    xxxxxxx
    >>> reg.predict(np.array([[3, 5]]))
    array([xx])
    """

    def __init__(self, model, kind, transform_params, model_params=None):
        
        if kind not in ['poly', 'cosine']:
            raise NotImplementedError

        self.kind = kind
        
        if self.kind == 'poly':
            if 'degree' not in transform_params.keys():
                raise ValueError('degree is a parameter for polynomial features')
            self._transform = PolynomialFeatures(degree=transform_params['degree'])

        elif self.kind == 'cosine':
            if 'degree' not in transform_params.keys():
                raise ValueError('degree is a parameter for cosine features')
            elif 'period' not in transform_params.keys():
                raise ValueError('period, is a parameter for cosine features')
            self._transform = CosineFeatures(degree=transform_params['degree'], T=transform_params['T'])
        
        self._params = transform_params
        self._model = model(**model_params)
        self._fitted = False
        self._phi_i = None
        
        

    def fit(self, X, y):
        if self._fitted:
            raise TypeError(f'The model {self.kind} is already fitted.')
        
        self._phi_i = self._transform.fit_transform(X)
        self._model.fit(self._phi_i, y)
        self._fitted = True

        return self

    def predict(self, X):
        if not self._fitted:
            raise TypeError(f"The model {self.kind} isn't fitted.")

        phi_predictor = self._transform.fit_transform(X)
        return self._model.predict(phi_predictor)


class BaseCV:
    """
    Base CrossValidation scheme for NonLinearRegression class
    """

    def __init__(self, model, num_models, criterion='mse'):

        if criterion == 'mse':
            self.criterion_name = criterion
            self.criterion = lambda y, y_pred: np.mean( (y-y_pred) ** 2)
        
        self.num_models = num_models
        self._fitted = False

        self.scores = np.zeros(num_models)
        self.scores_test = np.zeros(num_models)
        self.fitted_best_model = None
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        try:
            from tqdm import tqdm
            enum_models = enumerate(tqdm(self._models))
        
        except Exception as e:
            enum_models = enumerate(self._models)

        for idx, model in enum_models:
            model = model.fit(self.X_train, self.y_train)
        
            preds = model.predict(X)
            crit_res = self.criterion(y, preds)

            self.scores[idx] = crit_res
            # Too much mem used -> clear model, only keep scores
            self._models[idx] = None

        self._fitted = True
        return self
    
    def predict(self, X):
        if not self._fitted:
            raise TypeError(f"CV isn't fitted yet.")
        
        model = self.best_model()
        return  model.predict(X)
    
    def test_scores(self, X_test, y_test):
        raise NotImplementedError

        for idx, model in enumerate(self._models):
            preds = model.predict(X_test)
            crit_res = self.criterion(y_test, preds)

            self.scores_test[idx] = crit_res
        return self.scores_test
        
    def best_model(self):
        # Too much mem used -> clear model, only keep scores -> refit best model
        if self.fitted_best_model is None:
            print('Refitting best model...')
            idx_best = np.argmin(self.scores)
            self.fitted_best_model = self.model_used(
                model=self.base_model, 
                kind=self.kind, 
                transform_params=self.transforms_params[idx_best], 
                model_params=self.model_params).fit(self.X_train, self.y_train)

        return self.fitted_best_model

    def to_pickle(self, fn):
        with open(fn, 'wb') as f:
            idx_best = np.argmin(self.scores)

            self.best_params = self.transforms_params[idx_best]

            self.criterion = None
            pickle.dump(self, f)


class PolynomialRegressionCV(BaseCV):
    """
    Simple CrossValidation scheme for Polynomial Regression using a Scikit-Learn like base model.

    Parameters:
    ----------
    model : Scikit-Learn like object

    degrees : ndarray of shape (degrees,), default= (1.0, 2.0, 3.0)
        degrees values to test

    criterion : callable, string,  default='mse'
        objective function
    """

    def __init__(self, model, degrees=(1.0, 2.0, 3.0), criterion='mse', model_params=None):
        
        self.transforms_params = [{'degree': params} for params in degrees]
        super().__init__(
            model=model, num_models=len(self.transforms_params), criterion=criterion)

        self.base_model = model
        self.kind = 'poly'
        self.model_used = NonLinearRegression
        self._models = []
        self.model_params = model_params
        for transforms_params in self.transforms_params:
            non_linear_model = self.model_used(model=self.base_model, kind=self.kind, transform_params=transforms_params, model_params=self.model_params)
            self._models.append(non_linear_model)

class CosineRegressionCV(BaseCV):
    """
    Simple CrossValidation scheme for Cosine Regression using a Scikit-Learn like base model.

    Parameters:
    ----------
    model : Scikit-Learn like object

    degrees : ndarray of shape (degrees,), default= (1.0, 2.0, 3.0)
        degrees values to test
    
    periods : ndarray of shape (periods,), default= (1.0, 2.0, 3.0)
        degrees values to test

    criterion : callable, string,  default='mse'
        objective function
    """

    def __init__(self, model, degrees=(1.0, 2.0, 3.0), periods=(1.0, 2.0, 3.0), criterion='mse'):
        
        self.param_combinations = [list(zip(d, periods)) for d in itertools.permutations(degrees, len(periods))]
        self.transforms_params = [{'degree': params[0], 'period': params[1]} for params in self.param_combinations]

        super().__init__(
            model=model, num_models=len(self.transforms_params), criterion=criterion)
            
        self._models = []
        for degree in self.transforms_paramsx:
            model = NonLinearRegression(model=model, kind='cosine', *self.transforms_params)
            self._models.append(model)
