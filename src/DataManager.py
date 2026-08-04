from config import omegaconfig as conf


class DataManager:
    def __init__(self, dataframe=conf.get_global_conf().params.path_to_data):
        self.dataframe = dataframe

    def base_preprocess(self):
        pass

    def get_featured_data(self, data, feat_ver=conf.get_global_conf().params.feat_ver):

        pass
