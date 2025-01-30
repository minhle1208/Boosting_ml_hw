from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import optuna

from typing import Optional

def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])

class Boosting:
    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        subsample: int | float = 1.0,
        bagging_temperature: float = 1.0,
        bootstrap_type: str = 'Bernoulli',
        quantization_type: str | None = None,
        nbins: int = 255,
        rsm: float = 1.0
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.feat_masks: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.bootstrap_type = bootstrap_type
        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.quantization_type = quantization_type
        self.rsm = rsm
        self.nbins = nbins
        self.uniform_left = None
        self.uniform_right = None
        self.qbins = None
        self.feature_importances_ = None

        self.history = defaultdict(list) # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * (1 - self.sigmoid(y*z))  # Исправьте формулу на правильную. 

    def partial_fit(self, X, y, weight=None):
        return self.base_model_class(**self.base_model_params).fit(X, y, weight)

    def quantinize_uniform(self, X):
        return np.floor((X - self.uniform_left[None, :]) / (self.uniform_left - self.uniform_right)[None, :])
    
    def quantinize_quantile(self, X):
        X = X.copy()
        for feat in range(X.shape[1]):
            X[:, feat] = np.searchsorted(self.qbins[feat], X[:, feat])
        return X
    
    def quantinize(self, X):
        if self.quantization_type == 'Uniform':
            return self.quantinize_uniform(X)
        elif self.quantization_type == 'Quantile':
            return self.quantinize_quantile(X)
        return X

    def fit(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=None, plot=False, trial=None):
        """
        :param X_train: features array (train set)
        :param y_train: targets array (train set)
        :param X_val: features array (eval set)
        :param y_val: targets array (eval set)
        :param plot: bool 
        """

        N_train = X_train.shape[0]
        N_feat = X_train.shape[1]
        train_predictions = np.zeros(N_train)

        valid_flag = (X_val is not None) and (y_val is not None)
        if valid_flag:
            N_val = X_val.shape[0]
            val_predictions = np.zeros(N_val)

        # Quantization
        if self.quantization_type == 'Uniform':
            self.uniform_left = X_train.min(axis=0)
            self.uniform_right = X_train.max(axis=0)
                
        elif self.quantization_type == 'Quantile':
            self.qbins = []
            dist = np.linspace(0, N_train, self.nbins + 1).astype(int)[1:-1]
            for feat in range(X_train.shape[1]):
                srt = np.sort(X_train[:, feat])
                self.qbins += [srt[dist]]

        X_train = self.quantinize(X_train)
        if valid_flag:
            X_val = self.quantinize(X_val)

        for _ in range(self.n_estimators):
            # Features bootstrap
            feat_cnt = int(self.rsm * N_feat) if type(self.rsm) is float else self.rsm
            feat_mask = np.zeros(N_feat, dtype=np.bool_)
            feat_mask[np.random.permutation(N_feat)[:feat_cnt]] = True
            self.feat_masks += [feat_mask]

            # Bootstrap
            if self.bootstrap_type == 'Bernoulli':
                boot_sz = int(self.subsample * N_train) if type(self.subsample) is float else self.subsample
                boot_mask = np.zeros(N_train, dtype=np.bool_)
                boot_mask[np.random.permutation(N_train)[:boot_sz]] = True
                W = np.ones(N_train)
            elif self.bootstrap_type == 'Bayesian':
                boot_mask = np.ones(N_train, dtype=np.bool_)
                W = (-np.log(np.random.uniform(0, 1, N_train))) ** self.bagging_temperature * self.bagging_temperature
            X = X_train[boot_mask][:, feat_mask]
            y = y_train[boot_mask]
            W = W[boot_mask]

            grad = -self.loss_derivative(y, train_predictions[boot_mask])
            bmodel = self.partial_fit(X, grad, W)
            self.models += [bmodel]
            bpreds = bmodel.predict(X_train[:, feat_mask])
            g = self.find_optimal_gamma(y_train, train_predictions, bpreds)
            self.gammas += [g]
            train_predictions += self.learning_rate * g * bpreds
            self.history['train_loss'] += [self.loss_fn(y_train, train_predictions)]
            self.history['train_roc_auc'] += [roc_auc_score(y_train, self.sigmoid(train_predictions))]

            if valid_flag:
                val_predictions += self.learning_rate * g * bmodel.predict(X_val[:, feat_mask])
                self.history['val_loss'] += [self.loss_fn(y_val, val_predictions)]
                self.history['val_roc_auc'] += [roc_auc_score(y_val, self.sigmoid(val_predictions))]

                if early_stopping_rounds is not None and len(self.history['val_loss']) > early_stopping_rounds:
                    last = self.history['val_loss'][-early_stopping_rounds-1:]
                    if np.all(last == np.sort(last)):
                        break

                if trial is not None:
                    trial.report(self.history['valid_roc_auc'][-1], _)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

        self.feature_importances_ = np.zeros(N_feat)
        for model, feat_mask, g in zip(self.models, self.feat_masks, self.gammas):
            self.feature_importances_[feat_mask] += model.feature_importances_
        self.feature_importances_ /= self.feature_importances_.sum()

        if plot:
            self.plot_history(X_train, y_train, 'Train', False)
            if valid_flag:
                self.plot_history(X_val, y_val, 'Validation', False)

    def predict_proba(self, X):
        X = self.quantinize(X)
        predictions = np.zeros(X.shape[0])
        for model, feat_mask, g in zip(self.models, self.feat_masks, self.gammas):
            predictions += self.learning_rate * g * model.predict(X[:, feat_mask])
        result = self.sigmoid(predictions)
        return np.array([1 - result, result]).T

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, X, y):
        X = self.quantinize(X)
        return score(self, X, y)
        
    def plot_history(self, X, y, name='', quantinize=True):
        """
        :param X: features array (any set)
        :param y: targets array (any set)
        """
        if quantinize:
            X = self.quantinize(X)
        history = defaultdict(list)
        predictions = np.zeros(X.shape[0])
        for model, feat_mask, g in zip(self.models, self.feat_masks, self.gammas):
            predictions += self.learning_rate * g * model.predict(X[:, feat_mask])
            history['loss'] += [self.loss_fn(y, predictions)]
            history['roc_auc'] += [roc_auc_score(y, self.sigmoid(predictions))]

        _, axs = plt.subplots(1, 2)

        axs[0].plot(history['loss'])
        axs[0].set_xlabel('epoch')
        axs[0].set_ylabel('loss')
        axs[0].set_title(name)

        axs[1].plot(history['roc_auc'])
        axs[1].set_xlabel('epoch')
        axs[1].set_ylabel('ROC AUC')
        axs[1].set_title(name)

        plt.show()
