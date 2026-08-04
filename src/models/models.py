from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, \
    VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from config import omegaconfig as conf


def get_models():
    models = {
        'logistic_regression': LogisticRegression,
        'decision_tree': DecisionTreeClassifier,
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier,
        'knn': KNeighborsClassifier,
        'svc': SVC,
        'xgboost': XGBClassifier,
        'lightgbm': LGBMClassifier,
        'catboost': CatBoostClassifier,
    }
    return models


def get_ensembles_obj():
    stacking_cfg = conf.get_param_conf().models.stacking.versions.default
    voting_cfg = conf.get_param_conf().models.voting.versions.default

    base_estimators = [
        ('knn', KNeighborsClassifier(**dict(stacking_cfg.base_models.knn))),
        ('svc', SVC(**dict(stacking_cfg.base_models.svc))),
        ('tree', DecisionTreeClassifier(**dict(stacking_cfg.base_models.tree)))
    ]

    meta_model = LogisticRegression()

    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_model,
        cv=stacking_cfg.cv,
        stack_method='auto',
        n_jobs=-1
    )

    voting = VotingClassifier(
        estimators=base_estimators.copy(),
        voting=voting_cfg.voting,
        weights=None,
        n_jobs=-1
    )

    models = {"stack": stack, "voting": voting}
    return models
