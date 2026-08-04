from omegaconf import OmegaConf


def get_param_conf():
    return OmegaConf.load("src/config/param_config.yaml")


def get_global_conf():
    return OmegaConf.load("src/config/global_config.yaml")


