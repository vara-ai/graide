import bz2
import os
import pickle
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def file_exists(file_path: str) -> bool:
    return os.path.exists(file_path)


def create_directory(filepath: str, exist_ok: bool = True):
    """Creates a directory if it doesn't already exist."""

    path = os.path.abspath(filepath)
    if os.path.isdir(path) and exist_ok:
        return
    elif os.path.isfile(path):
        raise FileExistsError(f"Unable to create directory, {path} exists and is a file")
    os.makedirs(path, exist_ok=exist_ok)


def save_to_pickle(obj: object, filename: str, compress: bool = False, **kwargs):
    """Save a pickle-able object to a file. If the filename ends with .pklz save as a compressed pickle.

    Args:
        obj: The object we want to pickle.
        filename: The output filename of the pickled object.
        compress: an alternative mechanism to using a .pklz file extension for enabling bz2 compression
        **kwargs: Arguments passed to `pickle.dump`
    """

    create_directory(os.path.dirname(filename))

    if filename.endswith(".pklz") or compress:
        with bz2.BZ2File(filename, "w") as f:
            pickle.dump(obj, f, **kwargs)
    else:
        with open(filename, "wb") as f:
            pickle.dump(obj, f, **kwargs)


def load_from_pickle(filename: str, **kwargs) -> Any:
    """
    Loads a pickled object from a file.
    This function assumes the file is a BZ2 compressed pickle if the filename ends with `.pklz`.

    Args:
        filename: The filename of the pickled object to load.
        **kwargs: Arguments passed to `pickle.load`

    Returns:
        The unpickled object.
    """
    if filename.endswith(".pklz"):
        with bz2.BZ2File(filename, "r") as f:
            return pickle.load(f, **kwargs)
    else:
        with open(filename, "rb") as f:
            return pickle.load(f, **kwargs)


def save_df_as_csv(df: "pd.DataFrame", filename: str):
    """Saves a pandas dataframe as a CSV."""

    create_directory(os.path.dirname(filename))
    df.to_csv(filename)


def save_df_as_h5(df: "pd.DataFrame", filename: str):
    """Saves a pandas dataframe as an HDF5 file."""

    create_directory(os.path.dirname(filename))
    df.to_hdf(filename, key="data", format="table", encoding="utf-8")
