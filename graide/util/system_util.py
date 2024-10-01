from typing import Callable, Iterable, TypeVar

from joblib import delayed, Parallel
from tqdm import tqdm

CInput = TypeVar("CInput")
COutput = TypeVar("COutput")


def compute_parallel(
    computer: Callable[[CInput], COutput],
    arg_iter: Iterable[CInput],
    backend: str = "threading",
    n_jobs: int = -1,
    verbose: int = 0,
    tqdm_desc: str | None = None,
) -> list[COutput]:
    """
    Enables parallelising an arbitrary computation. The arg_iter is the list of input args whose computation should
    be parallelised, so this function only supports computer functions with a single argument.

    Args:
        computer: the function to run over multiple inputs in parallel.
        arg_iter:
            a list of args to pass to the computer function (=> for now we only support computer functions that
            expect a single argument).
        backend: see the joblib docs for more info.
        n_jobs: the maximum number of concurrently running jobs. If -1, all CPUs are used.
        verbose: verbosity level of the joblib logs
        tqdm_desc: string descriptor for the progress bar
    """

    args = list(arg_iter)
    return Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(computer)(a) for a in tqdm(args, total=len(args), desc=tqdm_desc)
    )
