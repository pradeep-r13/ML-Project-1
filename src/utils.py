import os
import sys
import dill

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):

    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(x_train, y_train, x_test, y_test, models, param):
    try:
        report = {}
        best_estimater = {}
        for name, model in models.items():
            para = param[name]

            grid = GridSearchCV(model, para, cv=3)
            grid.fit(x_train, y_train)

            best_model = grid.best_estimator_

            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            report[name] = test_score
            best_estimater[name] = best_model
        return best_estimater, report

    except Exception as e:
        raise CustomException(e, sys)
