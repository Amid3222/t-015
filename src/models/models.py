from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, \
    VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


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
        'stacking': StackingClassifier,
        'voting': VotingClassifier,
    }
    return models


