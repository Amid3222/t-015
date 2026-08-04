from sklearn.model_selection import KFold
from catboost import CatBoostClassifier, Pool
import numpy as np
from utils import utils
from config import omegaconfig as conf


class Validater:

    def k_fold_validate(self, dataframe, model_params, model_class, cat_features=None, kf=utils.get_skf()):
        X, y = utils.split_data_pd(dataframe, conf.get_global_conf().params.target_column_name)
        scores = []

        if model_class == CatBoostClassifier:
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                train_pool = Pool(X_train, y_train, cat_features=cat_features)
                val_pool = Pool(X_val, y_val, cat_features=cat_features)

                modelc = model_class
                model = modelc(**model_params)

                model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

                score = model.score(X_val, y_val)  # accuracy по умолчанию
                scores.append(score)
                print(f"Fold {fold + 1}: {score:.4f}")

            print(f"\nСредний score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
            return scores
        else:
            modelc = model_class
            model = modelc(**model_params)
            scores = utils.validate(model,
                                    utils.split_data_np(dataframe, conf.get_global_conf().params.target_column_name))
            print(f"\nСредний score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
            return scores

    def test_train_split_val(self, val_data_x, val_data_y, model):
        y_preds = model.predict(val_data_x)
        return utils.accuracy(y_preds, val_data_y)


    def create_model(self, model_class, model_params):
        modelc = model_class
        model = modelc(**model_params)
        return model
