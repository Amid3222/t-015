from catboost import CatBoostClassifier

from models import ModelsManager as mm
from training import Validater as v
from utils import utils
import numpy as np
import joblib
import DataManager


class PipelineRunner:

    def start_pipline(self):
        pass

    def run(self):
        d = DataManager.DataManager()
        data = d.get_featured_data(feat_ver=1)

        model_manager = mm.ModelsManager()
        validator = v.Validater()

        best_model_name = None
        best_cv_acc = 0
        best_model = None

        for model, model_name in model_manager.iterate_all_models().items():
            scores, best_fold_model = validator.k_fold(dataframe=data, model=model)

            utils.save_results_csv(model=model_name, params=model.get_params(deep=True), cv_median=np.mean(scores),
                                   cv_std=np.std(scores))

            if np.mean(scores) > best_cv_acc:
                best_cv_acc = np.mean(scores)
                best_model_name = model_name
                best_model = best_fold_model

        print(f"Best model: {best_model_name}, mean cv: {best_cv_acc}")
        joblib.dump(best_model, 'model.joblib')

    def evaluate_test_on_best(self):

        loaded_model = joblib.load('model.joblib')

        predictions = loaded_model.predict(X_test)
        pass
