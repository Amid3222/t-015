from pathlib import Path

from config import omegaconfig as conf
import pandas as pd
import re
import os
import numpy as np


class DataManager:
    def __init__(self, path=conf.get_global_conf().params.path_to_data):
        project_root = Path(__file__).resolve().parent.parent
        data_dir = project_root / "data"

        # self.original_df = pd.read_csv(f'{path}/train.csv')
        self.indexes = None
        # self.test_df = pd.read_csv(f'{path}/test.csv')
        self.original_df = {"train": pd.read_csv(data_dir / "train.csv"),

                            "test": pd.read_csv(data_dir / "test.csv")}

    def preprocess(self, mode_key="train"):
        data = self.original_df[mode_key]

        if mode_key == "test":
            self.indexes = data["PassengerId"]

        data = self.base_preprocess(data)
        data = self.get_featured_data(data)
        return data

    def base_preprocess(self, data):
        # self.indexes = self.original_df["PassengerId"]
        data = data.drop(columns=["Ticket", "Name", "Embarked", "PassengerId"])
        data = pd.get_dummies(data, columns=['Sex'], drop_first=True)
        data['Fare_log'] = np.log1p(data['Fare'])
        # self.original_df = self.original_df.Age.fillna(self.original_df.Age.mean())

        return data

    def get_featured_data(self, data, feat_ver=conf.get_global_conf().params.feat_ver):
        feat_df = data

        if feat_ver == 1:
            feat_df = feat_df.drop(columns=["Cabin"])
            return feat_df
        else:
            feat_df['Cabin'] = feat_df['Cabin'].apply(
                lambda x: re.sub(r'^([a-zA-Z]+).*', r'\1', x) if pd.notna(x) else 'Unknown')

            feat_df.loc[(feat_df["Cabin"] == "T") | (feat_df["Cabin"] == "G"), "Cabin"] = "Unknown"
            feat_df['FamilySize'] = feat_df['SibSp'] + feat_df['Parch'] + 1
            feat_df = pd.get_dummies(data=feat_df, columns=["Cabin"])

            return feat_df

    def get_df_info(self):
        print(self.original_df.shape)
        print(self.original_df.info())
        print(self.original_df.describe())
        print(self.original_df.isnull().sum(axis=0))

    def fillnas(self, X_train: pd.DataFrame, X_val: pd.DataFrame, columns):
        fill_dict = {}

        for col in columns:
            fill_dict[col] = X_train[col].median(skipna=True)

        X_train.fillna(fill_dict, inplace=True)
        X_val.fillna(fill_dict, inplace=True)

        return X_val, X_train
