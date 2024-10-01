"""
This is a plugin to allow Hydra to find the conf directory (graide/conf).
"""
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

from graide.util.path_util import PACKAGE_ROOT


class VaraSearchpathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.prepend(
            provider="vara-searchpath-plugin",
            path=f"file://{PACKAGE_ROOT}/graide/conf",
        )
