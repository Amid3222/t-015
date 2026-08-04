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

        for model_info in model_manager.iter_all_models(create_model_obj=False):
            model_class = model_info.model_class

            for params_ver in model_info.params_keys:

                params = model_manager.get_model_params_by_key(model_info.model_name, param_key=params_ver)

                scores, best_model = validator.k_fold_validate(dataframe=data, model_params=params,
                                                               model_class=model_class)

                utils.save_results_csv(model=model_info.model_name, param_ver=params_ver, cv_median=np.mean(scores),
                                       cv_std=np.std(scores))

                if np.mean(scores) > best_cv_acc:
                    best_cv_acc = np.mean(scores)
                    best_model_name = model_info.model_name + " " + params_ver
                    best_model = model_class(**params)

        print(f"Best model: {best_model_name}, mean cv: {best_cv_acc}")
        joblib.dump(best_model, 'model.joblib')

    def evaluate_test_on_best(self):

        loaded_model = joblib.load('model.joblib')

        predictions = loaded_model.predict(X_test)
        pass
