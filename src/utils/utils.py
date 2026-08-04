from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV



def validate(model, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X, y, cv=skf)

    return score

def save_results_csv(path_f, **kwargs):
    print(f"saved {kwargs}")
    pass


def get_skf(splits=5, r_state=42):
    return StratifiedKFold(n_splits=splits, shuffle=True, random_state=r_state)


def split_data_pd(data, target: str):
    X_data = data.drop(columns=target)
    y_data = data[target]

    return X_data, y_data


def split_data_np(data, target: str):
    X_data = data.drop(columns=target).values
    y_data = data[target].values

    return X_data, y_data
