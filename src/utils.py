import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, ParameterGrid
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj, compress=3):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        joblib.dump(obj, file_path, compress=compress)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        return joblib.load(file_path)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models, param, n_iter=20, cv=3, random_state=42):
    try:
        report = {}

        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]

            if para:
                # n_iter can't exceed the total number of possible combinations
                n_combinations = len(ParameterGrid(para))
                search_n_iter = min(n_iter, n_combinations)

                rs = RandomizedSearchCV(
                    model,
                    param_distributions=para,
                    n_iter=search_n_iter,
                    cv=cv,
                    random_state=random_state,
                    n_jobs=-1
                )
                rs.fit(X_train, y_train)

                logging.info(f"Best parameters for {model_name}: {rs.best_params_}")
                model.set_params(**rs.best_params_)
            else:
                logging.info(f"No hyperparameter grid for {model_name}, using defaults")

            model.fit(X_train, y_train)  # Train model with best (or default) parameters

            # Predict Testing data
            y_test_pred = model.predict(X_test)

            test_model_score = r2_score(y_test, y_test_pred)

            logging.info(f"{model_name}: {test_model_score}")

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)