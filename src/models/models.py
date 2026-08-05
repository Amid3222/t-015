from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
    m = conf.get_param_conf().models

    return {
        "logistic_regression+baseline": LogisticRegression(**m.logistic_regression.versions.baseline),
        "logistic_regression+l2": LogisticRegression(**m.logistic_regression.versions.l2),
        "logistic_regression+l1": Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(**m.logistic_regression.versions.l1)),
        ]),
        "logistic_regression+elasticnet": LogisticRegression(**m.logistic_regression.versions.elasticnet),

        "knn+default": KNeighborsClassifier(**m.knn.versions.default),

        "decision_tree+default": DecisionTreeClassifier(**m.decision_tree.versions.default),

        "random_forest+best": RandomForestClassifier(**m.random_forest.versions.best),

        "catboost+default": CatBoostClassifier(**m.catboost.versions.default),

        "xgboost+best": XGBClassifier(**m.xgboost.versions.best),

        "lightgbm+default": LGBMClassifier(**m.lightgbm.versions.default),

        "stacking+default": StackingClassifier(
            estimators=[
                ("knn", KNeighborsClassifier(**m.stacking.versions.default.base_models.knn)),
                ("svc", SVC(**m.stacking.versions.default.base_models.svc)),
                ("tree", DecisionTreeClassifier(**m.stacking.versions.default.base_models.tree)),
            ],
            final_estimator=LogisticRegression(),
            cv=m.stacking.versions.default.cv,
        ),

        "voting+default": VotingClassifier(
            estimators=[
                ("logistic_regression", LogisticRegression()),
                ("knn", KNeighborsClassifier()),
                ("svc", SVC(probability=True)),
            ],
            voting=m.voting.versions.default.voting,
        ),
    }
