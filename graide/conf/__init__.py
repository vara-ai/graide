import warnings

# filter because Pydantic dataclasses don't understand '_target_' => lots of noise
warnings.filterwarnings("ignore", message="fields may not start with an underscore")

# Needed such that these configs are always registered (i.e. by anything that imports from the `conf` module).
from graide.conf.strategy import *  # noqa
