import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor)

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join('artifacts', 'model.pkl')


class Modeltrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Spliting Training and Test input Data')
            x_train, x_test, y_train, y_test = (
                train_array[:, :-1], test_array[:, :-1], train_array[:, -1], test_array[:, -1])

            models = {
                'Random Forest Regressor': RandomForestRegressor(),
                'AdaBoost Regressor': AdaBoostRegressor(),
                'GradientBoosting Regressor': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'SVR': SVR(),
                'K-Neighbors Regressor': KNeighborsRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'XGBRegressor': XGBRegressor(),
                'CatBoostRegressor': CatBoostRegressor(verbose=False)
            }

            

            params = {

                "Random Forest Regressor": {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },

                "AdaBoost Regressor": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.5]
                },

                "GradientBoosting Regressor": {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5],
                    'subsample': [0.8, 1.0]
                },

                "Linear Regression": {
                    # no hyperparameters
                },

                "SVR": {
                    'kernel': ['rbf', 'linear'],
                    'C': [1, 10],
                    'epsilon': [0.1, 0.2],
                    'gamma': ['scale', 'auto']
                },

                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },

                "DecisionTreeRegressor": {
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },

                "XGBRegressor": {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                },

                "CatBoostRegressor": {
                    'iterations': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'depth': [6, 8],
                    'l2_leaf_reg': [1, 3]
                }
            }

            

            best_estimater, model_report = evaluate_models(
                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models, param=params)

            # To get best model score from dict

            best_model_score = max(model_report.values())

            # To get best model name from dict

            best_model_name = list(model_report.keys())[list(
                model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            best_estimater = best_estimater[best_model_name]
            if best_model_score < 0.6:
                raise CustomException('No Best Model Found', sys)

            logging.info(
                'Best Model Found On Both Training And Testing Dataset')

            save_object(
                file_path=self.model_trainer_config.train_model_file_path, obj=best_model)
            logging.info('save_object')
            Predicted = best_estimater.predict(x_test)

            r2_square = r2_score(y_test, Predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
