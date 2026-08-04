from models import ModelsManager as mm
from training import Validater as v
from utils import utils
import numpy as np


class Trainer:

    def run(self):
        model_manager = mm.ModelsManager()
        validator = v.Validater()

        best_model_name = None
        best_cv_acc = 0
        best_model = None

        for model_info in model_manager.iter_all_models(create_model_obj=False):
            model_class = model_info.model_class

            for params_ver in model_info.params_keys:
                params = model_manager.get_model_params_by_key(model_info.model_name, param_key=params_ver)

                scores = validator.k_fold_validate(dataframe=None, model_params=params, model_class=model_class)
                utils.save_results_csv(model=model_info.model_name, param_ver=params_ver, cv_median=np.mean(scores),
                                       cv_std=np.std(scores))

                if np.mean(scores) > best_cv_acc:
                    best_cv_acc = np.mean(scores)
                    best_model_name = model_info.model_name + " " + params_ver
                    best_model = model_class(**params)

        print(f"Best model: {best_model_name}, mean cv: {best_cv_acc}")
