from config import omegaconfig as conf
import models
from omegaconf import OmegaConf


class ModelsManager:
    def __init__(self, config_path=None):
        self.model_reg = models.get_models()
        self.param_config = conf.get_param_conf() if config_path is None else OmegaConf.load(config_path)

    def get_model_from_config(self, model_name: str):
        models = dict()
        param_versions = self.param_config.models[model_name].versions.keys()

        for version_name in param_versions:
            params = OmegaConf.to_container(self.param_config.models[model_name].versions[version_name])
            model_class = self.model_reg[model_name]
            models[version_name](model_class(**params))

        return models

    def get_model_reg(self):
        return self.model_reg

    def iter_all_models(self):
        for model_name in self.model_reg.keys():
            yield self.get_model_from_config(model_name)
