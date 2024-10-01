import copy
import datetime
import os
import subprocess
import uuid
from typing import Callable, Type, TypeVar

from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import OmegaConf
from pydantic.dataclasses import dataclass as pydantic_dataclass

CONFIG_STORE = ConfigStore.instance()  # Hydra ConfigStore singleton


def register_config(group: str | None = None) -> Callable:
    """Decorator for registering structured configs (dataclasses) with Hydra"""

    def decorator_config_class(config_class: Type) -> Type:
        if group is not None:
            CONFIG_STORE.store(name=config_class.__name__, node=config_class, group=group)
        return config_class

    return lambda config_class: decorator_config_class(pydantic_dataclass(config_class))


@register_config()
class BaseConfig:
    ...


# Type for annotations that allow BaseConfig and all subclasses
BaseConfigOrSubType = TypeVar("BaseConfigOrSubType", bound=BaseConfig)


def _get_commit_sha() -> str:
    try:
        return subprocess.getoutput("git rev-parse --short HEAD").rstrip()
    except subprocess.CalledProcessError:
        return ">no git"


def save_config(save_dir: str, config: BaseConfig) -> str:
    """Saves a single config as a YAML file within a specified output directory.

    Args:
        save_dir: The directory in which to save the config as YAML.
        config: A config object to save as YAML.

    Returns: the path to the saved YAML.
    """
    filename_parts = [
        "config",
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        uuid.uuid4().hex[:4],
    ]
    filename = "-".join(filename_parts) + ".yaml"
    dest_path = os.path.join(save_dir, filename)

    to_save = OmegaConf.merge({"commit_sha": _get_commit_sha()}, config)

    with open(dest_path, "w") as h:
        h.write(OmegaConf.to_yaml(to_save, sort_keys=False, resolve=True))

    return dest_path


def compose_one(config_group: str, config_name: str, instantiate: bool = False):
    """Returns a config that has already been registered to the CONFIG_STORE."""

    # We copy here s.t. applying the overrides doesn't modify the stored config
    config = copy.deepcopy(CONFIG_STORE.repo[config_group][f"{config_name}.yaml"].node)

    if instantiate:
        return hydra_instantiate(config)

    return config
