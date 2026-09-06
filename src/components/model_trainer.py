from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Support Vector Regressor": SVR(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params = {
                "Linear Regression": {},

                "Lasso": {
                    'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                    'selection': ['cyclic', 'random']
                },

                "Ridge": {
                    'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr']
                },

                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]
                },

                "Decision Tree": {
                    'criterion': ['squared_error', 'absolute_error', 'poisson'],
                    'max_depth': [5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                },

                "Random Forest Regressor": {
                    'n_estimators': [50, 75, 100],
                    'max_depth': [8, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'max_features': ['sqrt', 'log2']
                },

                "Support Vector Regressor": {
                    'kernel': ['linear', 'rbf', 'poly'],
                    'C': [0.1, 1, 10, 100],
                    'epsilon': [0.01, 0.1, 0.5, 1],
                    'gamma': ['scale', 'auto']
                },

                "AdaBoost Regressor": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0],
                    'loss': ['linear', 'square', 'exponential']
                },
            }

            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models, param=params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            else:
                best_model = models[best_model_name]
                logging.info(f"Best found model on both training and testing dataset: {best_model_name} with r2 score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            model_size_mb = os.path.getsize(self.model_trainer_config.trained_model_file_path) / (1024 * 1024)
            logging.info(f"Saved model size: {model_size_mb:.2f} MB")

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            logging.info(f"R2 Score: {r2_square}")
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)