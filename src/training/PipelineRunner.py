from idlelib import rpc
from sys import prefix

import pandas as pd
from catboost import CatBoostClassifier
from config import omegaconfig as conf
from models import ModelsManager as mm
from training import Validater as v
from utils import utils
import numpy as np

import joblib
import DataManager


class PipelineRunner:

    def __init__(self):
        self.data_manager = DataManager.DataManager()

    def start_pipline(self):
        print("pipline started")
        self._run()
        # self.evaluate_best()
        # self.create_submission()
        # self.show_model_stats()

    def _run(self):
        d = self.data_manager
        data = d.preprocess(mode_key="train")

        model_manager = mm.ModelsManager()
        validator = v.Validater()

        best_model_name = None
        best_cv_acc = 0
        best_model = None

        print("start k-fold process...")

        for model, model_name in model_manager.iterate_all_models().items():

            if str(model_name).startswith("catboost"):
                validator.k_fold_catboost(dataframe=data, model_params=model.get_params(deep=True),
                                          model_class=model.__class__)
                print(f"validating model {model_name} with score {np.mean(scores)}")

            scores, best_fold_model = validator.k_fold(dataframe=data, model=model)
            print(f"validating model {model_name} with score {np.mean(scores)}")

            utils.save_results_csv(model=model_name, params=model.get_params(deep=True), cv_median=np.mean(scores),
                                   cv_std=np.std(scores))

            print(f"model {model_name} saved to csv")

            if np.mean(scores) > best_cv_acc:
                best_cv_acc = np.mean(scores)
                best_model_name = model_name
                best_model = best_fold_model
                print(f"Best model updated {best_model_name}, mean cv: {best_cv_acc}")

        print(f"Best model: {best_model_name}, mean cv: {best_cv_acc}")
        joblib.dump(best_model, 'model.joblib')

    def _create_submission(self):
        print("creating test submission on best model")
        d = DataManager.DataManager()
        data = d.preprocess(mode_key="test")
        loaded_model = joblib.load('model.joblib')
        predictions = loaded_model.predict(data)
        sub = pd.concat([d.indexes, predictions])
        sub.to_csv('submisson.csv')
        print(f"Submission saved as submisson.csv")

    def _evaluate_best(self):
        loaded_model = joblib.load('model.joblib')

        df = self.data_manager.preprocess()
        target = conf.get_global_conf().params.target_column_name

        X_train, X_test, y_train, y_test = utils.test_train_split(utils.split_data_np(df, target))

        validator = v.Validater()
        acc = validator.test_train_split_val(X_test, y_test, loaded_model)
        print(f"Val accuracy on best model {acc}")

    def _show_model_stats(self):
        df = pd.read_csv(conf.get_global_conf().save_result_to)
        print("Models statistic")
        print(df)
