from config import omegaconfig as conf
import models
from omegaconf import OmegaConf
from dataclass import dataclasses


class ModelsManager:
    def __init__(self, config_path=None):
        self.model_reg = models.get_models()
        self.param_config = conf.get_param_conf() if config_path is None else OmegaConf.load(config_path)

    def iterate_all_models(self):
        for model_name, model in self.model_reg.items():
            yield model, model_name

    def get_model_by_name(self, name:str):
        self.model_reg[name]


