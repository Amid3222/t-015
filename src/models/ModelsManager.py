from config import omegaconfig as conf
import models
from omegaconf import OmegaConf
from dataclass import dataclasses


class ModelsManager:
    def __init__(self, config_path=None):
        self.model_reg = models.get_models()
        self.param_config = conf.get_param_conf() if config_path is None else OmegaConf.load(config_path)

    def get_model_class_from_config(self, model_name: str):

        param_versions = self.param_config.models[model_name].versions.keys()
        params_keys = []
        model_class = self.model_reg[model_name]

        for version_name in param_versions:
            params_keys.append(version_name)

        data = dataclasses.ModelClassInfo(model_name=model_name, model_class=model_class, params_keys=param_versions)
        return data

    def get_model_objs_from_config(self, model_name: str):
        '''
        {"model_name+ver": model_obj}
        '''
        model_data = self.get_model_class_from_config(model_name)

        models = []

        for param_key in model_data["key"]:
            model = self.create_model_obj(model_data[model_name], self.get_model_params_by_key(model_name, param_key))
            model = dataclasses.ModelObj(model_name=model_name, model=model, param_ver=param_key)
            models.append(model)
        return models

    def get_model_params_by_key(self, model_name, param_key):
        params = OmegaConf.to_container(self.param_config.models[model_name].versions[param_key])
        return params

    def get_model_registry(self):
        return self.model_reg

    def iter_all_models(self, create_model_obj=False):
        m_names = self.get_model_registry()

        for m_name, m_class in m_names.items():
            if create_model_obj:
                yield self.get_model_objs_from_config(m_name)
            else:
                yield self.get_model_class_from_config(m_name)

    def create_model_obj(self, modelclass, params):
        return modelclass(**params)
