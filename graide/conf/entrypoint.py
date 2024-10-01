import functools
import inspect
import os
from typing import Any, Callable

import hydra
from omegaconf import OmegaConf

from graide.conf.lib import BaseConfigOrSubType
from graide.util import io_util
from graide.util.logging import logger


def validate_config(
    entrypoint_function: Callable[[Any], Any],
    cache_artefacts_func: Callable[[BaseConfigOrSubType], str] | None = None,
):
    """A decorator that validates an entrypoint config according to the type that that config was
    annotated with as well as checks whether the given entrypoint's cached artefacts already exist.

    Args:
        entrypoint_function:
            The entrypoint function that was decorated with `@entrypoint(...)`. The `validated_config_entrypoint`
            function wrapper below has the same type signature as the `entrypoint_function` this decorator decorates.
        cache_artefacts_func:
            A function that returns a path indicating the file / directory that represents the entrypoint's cache
            (i.e. if that file / directory exists, the entrypoint will not be run).
    """

    @functools.wraps(entrypoint_function)
    def validated_config_entrypoint(config: BaseConfigOrSubType):

        # Trigger dynamic config validation (e.g. `MISSING` check)
        OmegaConf.to_object(config)

        # Trigger interpolations to make sure all can be resolved, before deploying to a remote machine
        OmegaConf.resolve(config)

        # If there is a cache artefact path to check, and if it exists, don't re-run the entrypoint.
        if cache_artefacts_func is not None:
            cache_dir = cache_artefacts_func(config)
            if io_util.file_exists(cache_dir):
                entrypoint_file = os.path.basename(inspect.getabsfile(entrypoint_function))
                logger.info(f"Artefacts for `{entrypoint_file}` are cached at '{cache_dir}', => exiting!")
                return

        return entrypoint_function(config)

    return validated_config_entrypoint


def entrypoint(
    config_path: str,
    config_name: str,
    cache_artefacts_func: Callable[[BaseConfigOrSubType], str] | None = None,
):
    """
    A decorator that implements custom dynamic config validation (using Pydantic) as well as cache validation before
    passing the resulting config on to Hydra's main decorator.

    Args:
        config_path: the path to the directory in which the entrypoint YAML is located
        config_name: the name of the entrypoint YAML file (excluding `.yaml`)
        cache_artefacts_func:
            A function that takes a given entrypoint config and returns the path where results should be cached.
    """

    def entrypoint_decorator(entrypoint_function: Callable[[Any], Any]):
        hydra_decorator = hydra.main(config_path=config_path, config_name=config_name, version_base="1.2")
        return hydra_decorator(validate_config(entrypoint_function, cache_artefacts_func=cache_artefacts_func))

    return entrypoint_decorator
