import dataclasses
from typing import Callable, Protocol

import numpy as np
from scipy import interpolate
from sklearn.metrics import auc, roc_curve

from graide.util.types import BooleanArray, PermTestResult, ValueWithCI


class MetricFunc(Protocol):
    """
    A MetricFunc has three required arguments — `y_true`, `y_pred`, & `weights` — and one optional argument — `axis` —
    and returns either a single float or an array of floats depending on the dimensions of the input arrays. The
    functions with this signature below are compute_[sens/spec/cdr/recall_rate].
    """

    def __call__(
        self,
        y_true: BooleanArray,
        y_pred: BooleanArray,
        weights: np.ndarray,
        axis: int | None = None,
    ) -> float | np.ndarray:
        ...


def _repeat_or_flatten(arr: np.ndarray, n_repeats: int) -> np.ndarray:
    """
    This function creates an array of shape `(len(arr) * n_repeats,)`, either via repeating the input array `n_repeats`
    times (if it only has a single dimension) or via flattening the array if it's already the correct shape.
    """

    assert arr.ndim == 1 or arr.shape[1] == n_repeats, (
        "Your input array must either have only a single dimension or have the correct number of repeats in the "
        "second dimension."
    )

    return np.repeat(arr, n_repeats, axis=0) if arr.ndim == 1 else arr.flatten()


def _align_shapes_and_flatten(ref_arr: np.ndarray, *other_arrs) -> tuple[np.ndarray, ...]:
    """
    Ensures that all arrays in the `other_arrs` list have the same shape as `ref_arr`, either (n_studies,) or
    (n_studies, n_preds), and then that all arrays are flattened to be 1D.
    """

    if ref_arr.ndim > 1:
        n_repeats = ref_arr.shape[1]
        other_arrs = [_repeat_or_flatten(arr, n_repeats) for arr in other_arrs]
    else:
        assert all(arr.shape == ref_arr.shape for arr in other_arrs)

    return ref_arr.flatten(), *other_arrs


def compute_sens(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    axis: int | None = None,
) -> float | np.ndarray:
    """
    Computes sensitivity as the weighted sum of true positives over positives.

    Each input array can either be of shape (n_samples,) or (n_resamples, n_samples), the former leading to
    a standard scalar calculation and the latter leading to a vectorised calculation over resamples.

    Args:
        y_true: boolean ground-truth labels
        y_pred: boolean predictions
        weights: per-sample (float) weights
        axis: set to 1 / -1 if metric aggregation over n_samples should happen in parallel for n_resamples

    Returns:
        Either a scalar sensitivity value or an (n_resamples,) vector of sensitivities for each resample (where
        aggregation is performed over axis=1).
    """

    numerator = ((y_true & y_pred) * weights).sum(axis)
    denominator = (y_true * weights).sum(axis)

    if axis is None and (numerator == 0 and denominator == 0):
        return 1.0

    return numerator / denominator


def compute_spec(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    axis: int | None = None,
) -> float | np.ndarray:
    """
    Computes specificity as the weighted sum of true negatives over negatives.

    Each input array can either be of shape (n_samples,) or (n_resamples, n_samples), the former leading to
    a standard scalar calculation and the latter leading to a vectorised calculation over resamples.

    Args:
        y_true: boolean ground-truth labels
        y_pred: boolean predictions
        weights: per-sample (float) weights
        axis: set to 1 / -1 if metric aggregation over n_samples should happen in parallel for n_resamples

    Returns:
        Either a scalar specificity value or an (n_resamples,) vector of specificities for each resample (where
        aggregation is performed over axis=1).
    """
    return ((~y_true & ~y_pred) * weights).sum(axis) / (~y_true * weights).sum(axis)


def compute_cdr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    axis: int | None = None,
) -> float | np.ndarray:
    """Computes CDR as the weighted sum of _correct_ positive predictions over all predictions.

    Each input array can either be of shape (n_samples,) or (n_resamples, n_samples), the former leading to
    a standard scalar calculation and the latter leading to a vectorised calculation over resamples.

    Args:
        y_true: boolean ground-truth labels
        y_pred: boolean predictions
        weights: per-sample (float) weights
        axis: set to 1 / -1 if metric aggregation over n_samples should happen in parallel for n_resamples

    Returns:
        Either a scalar CDR or an (n_resamples,) vector of CDRs for each resample (where aggregation
        is performed over axis=1).
    """

    return ((y_pred & y_true) * weights).sum(axis=axis) / weights.sum(axis=axis)


def compute_recall_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    axis: int | None = None,
) -> float | np.ndarray:
    """Computes recall rate as the weighted sum of positive predictions over all predictions.

    Each input array can either be of shape (n_samples,) or (n_resamples, n_samples), the former leading to
    a standard scalar calculation and the latter leading to a vectorised calculation over resamples.

    Args:
        y_true: boolean ground-truth labels, not used here but required to match the `metric_fn` callable type
        y_pred: boolean predictions
        weights: per-sample (float) weights
        axis: set to 1 / -1 if metric aggregation over n_samples should happen in parallel for n_resamples

    Returns:
        Either a scalar recall rate or an (n_resamples,) vector of recall rates for each resample
        (where aggregation is performed over axis=1).
    """

    return (y_pred * weights).sum(axis=axis) / weights.sum(axis=axis)


def one_hot(tensor: np.ndarray, num_classes: int = -1, dtype="int"):
    """
    Takes a tensor with index values of shape ``(*)`` and returns a tensor of shape ``(*, num_classes)``
    that have zeros everywhere except where the index of last dimension matches the corresponding value
    of the input tensor, in which case it will be 1.

    See also `One-hot on Wikipedia` [https://en.wikipedia.org/wiki/One-hot]

    Arguments:
        tensor: integer-typed, class values of any shape
        num_classes:
            Total number of classes. If -1 then the number of classes will be inferred as one greater than the
            largest class value in the input tensor.
        dtype: the dtype of the output tensor

    Returns:
        An array that has one more dimension with 1 values at the index of last dimension indicated by the input,
        and 0 everywhere else.

    Examples:
        >>> one_hot(np.array(range(0, 5)) % 3)
        array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0]])
        >>> one_hot(np.array(range(0, 5)) % 3, num_classes=5)
        array([[1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0]])
    """

    assert tensor.dtype == int, "Your input tensor must have the integer dtype"

    if num_classes == -1:
        num_classes = np.max(tensor) + 1
    output_shape = tensor.shape + (num_classes,)

    one_hot = np.zeros((tensor.size, num_classes), dtype=dtype)
    one_hot[np.arange(tensor.size), tensor.reshape(-1)] = 1
    one_hot = np.reshape(one_hot, output_shape)

    return one_hot


def generate_resamples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    n_resamples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper function to generate bootstrapping samples, with replacement.

    Args:
        y_true: (n_samples,) boolean ground-truth labels
        y_pred: (n_samples,) or (n_samples, n_pred) boolean predictions
        weights: (n_samples,) per-sample (float) weights
        n_resamples: the number of samples with replacement used to construct bootstrap CIs

    Returns:
        y_true_resampled (n_resamples, n_samples)
        y_pred_resampled (n_resamples, n_samples)
            if multiple predictions per sample are provided, pooling happens before every resampling step
        weights_resampled (n_resamples, n_samples)
    """

    # Bootstrap resamples
    n_samples = len(y_true)
    rng = np.random.default_rng()  # Faster than np.random.randint
    resamples = rng.integers(low=0, high=n_samples, size=(n_resamples, n_samples))

    # (n_resamples, n_samples)
    y_true_resampled = y_true[resamples]
    weights_resampled = weights[resamples]

    if y_pred.ndim == 2:
        n_pred = y_pred.shape[1]
        assert y_pred.shape == (
            n_samples,
            n_pred,
        ), f"Inconsistent shape of y_pred {y_pred.shape}, expecting {(n_samples, n_pred)}."
        # (n_resamples, n_samples, n_pred)

        pred_samples = one_hot(
            tensor=np.random.randint(low=0, high=n_pred, size=(n_resamples, n_samples)),
            num_classes=n_pred,
            dtype=y_true.dtype,
        )
        if n_samples == 1:
            # Edge case for local data/small samples: Keras' `to_categorical` is removing trailing singleton dimensions
            # because it is usually meant to convert from an integer class vector to a binary class matrix.
            pred_samples = pred_samples[:, None, :]
        assert pred_samples.shape == (n_resamples, n_samples, n_pred)
        y_pred_resampled = np.array(
            [y_pred[pred_sample][resample] for pred_sample, resample in zip(pred_samples, resamples)]
        )

    else:
        y_pred_resampled = y_pred[resamples]

    return y_true_resampled, y_pred_resampled, weights_resampled


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
) -> tuple[ValueWithCI, ValueWithCI, ValueWithCI, ValueWithCI]:
    """
    Computes sensitivity, specificity, CDR, & recall rate, all with confidence intervals.

    NB: If `y_pred` contains more than one (n) prediction for each sample, we'll compute average metrics via flattening
    the predictions into a 1D array as if we had n times as many samples (y_true & weights will be duplicated to match).

    Args:
        y_true: (n_samples,) boolean ground-truth labels
        y_pred: (n_samples,) or (n_samples, n_pred), boolean predictions
        weights: (n_samples,) per sample (float) weights

    Returns: A `ValueWithCI` instance for each of sens, spec, CDR, & recall rate
    """

    y_pred, y_true, weights = _align_shapes_and_flatten(y_pred, y_true, weights)

    # Compute _scalar_ metrics
    sens = compute_sens(y_true, y_pred, weights)
    spec = compute_spec(y_true, y_pred, weights)
    cdr = compute_cdr(y_true, y_pred, weights)
    recall_rate = compute_recall_rate(y_true, y_pred, weights)

    # Compute CIs for each metric via resampling
    if n_resamples == 0:
        sens_with_ci = ValueWithCI(value=sens, ci_low=None, ci_upp=None)
        spec_with_ci = ValueWithCI(value=spec, ci_low=None, ci_upp=None)
        cdr_with_ci = ValueWithCI(value=cdr, ci_low=None, ci_upp=None)
        recall_rate_with_ci = ValueWithCI(value=recall_rate, ci_low=None, ci_upp=None)

    else:

        y_true_resampled, y_pred_sampled, weights_resampled = generate_resamples(y_true, y_pred, weights, n_resamples)

        sens_resampled = compute_sens(y_true_resampled, y_pred_sampled, weights_resampled, axis=1)
        sens_with_ci = ValueWithCI(
            value=sens,
            ci_low=np.quantile(sens_resampled, alpha / 2.0),
            ci_upp=np.quantile(sens_resampled, 1 - alpha / 2.0),
        )

        spec_resampled = compute_spec(y_true_resampled, y_pred_sampled, weights_resampled, axis=1)
        spec_with_ci = ValueWithCI(
            value=spec,
            ci_low=np.quantile(spec_resampled, alpha / 2.0),
            ci_upp=np.quantile(spec_resampled, 1 - alpha / 2.0),
        )

        cdr_resampled = compute_cdr(y_true_resampled, y_pred_sampled, weights_resampled, axis=1)
        cdr_with_ci = ValueWithCI(
            value=cdr,
            ci_low=np.quantile(cdr_resampled, alpha / 2.0),
            ci_upp=np.quantile(cdr_resampled, 1 - alpha / 2.0),
        )

        recall_rate_resampled = compute_recall_rate(y_true_resampled, y_pred_sampled, weights_resampled, axis=1)
        recall_rate_with_ci = ValueWithCI(
            value=recall_rate,
            ci_low=np.quantile(recall_rate_resampled, alpha / 2.0),
            ci_upp=np.quantile(recall_rate_resampled, 1 - alpha / 2.0),
        )

    return sens_with_ci, spec_with_ci, cdr_with_ci, recall_rate_with_ci


def compute_stratified_sens(
    stratifications: dict[str, BooleanArray],
    y_true: BooleanArray,
    y_pred: BooleanArray,
    weights: np.ndarray,
    n_resamples: int = 0,
    alpha: float = 0.05,
) -> dict[str, ValueWithCI]:
    """Calls `compute_sens` for each stratification (mask) passed."""

    n_repeats = 1 if y_pred.ndim == 1 else 2
    y_pred, y_true, weights = _align_shapes_and_flatten(y_pred, y_true, weights)

    stratified_sens = {}
    for strat_name, mask in stratifications.items():
        mask = _repeat_or_flatten(mask, n_repeats)
        sens_value = compute_sens(y_pred[mask], y_true[mask], weights[mask])

        if n_resamples == 0:
            sens_with_ci = ValueWithCI(value=sens_value, ci_low=None, ci_upp=None)
        else:
            y_true_resampled, y_pred_sampled, weights_resampled = generate_resamples(
                y_true[mask], y_pred[mask], weights[mask], n_resamples
            )
            sens_resampled = compute_sens(y_true_resampled, y_pred_sampled, weights_resampled, axis=1)
            sens_with_ci = ValueWithCI(
                value=sens_value,
                ci_low=np.quantile(sens_resampled, alpha / 2.0),
                ci_upp=np.quantile(sens_resampled, 1 - alpha / 2.0),
            )

        stratified_sens[strat_name] = sens_with_ci

    return stratified_sens


def compute_stratified_cdr(
    stratifications: dict[str, BooleanArray],
    y_true: BooleanArray,
    y_pred: BooleanArray,
    weights: np.ndarray,
    n_resamples: int = 0,
    alpha: float = 0.05,
) -> dict[str, ValueWithCI]:
    """Calls `compute_cdr` for each stratification passed."""

    n_repeats = 1 if y_pred.ndim == 1 else 2
    y_pred, y_true, weights = _align_shapes_and_flatten(y_pred, y_true, weights)

    def _compute_cdr(
        y_pred_m: np.ndarray,
        y_true_m: np.ndarray,
        weights_m: np.ndarray,
        axis: int | None = None,
    ) -> float | np.ndarray:
        """
        An alternative version of the `compute_cdr` function that always uses the total weight in the
        denominator, i.e. so these CDRs make sense as fractions of the CDR of the unstratified data.
        NB: the inputs here are suffixed with `_m` whereas the overall arrays (e.g. 'weights') are not.
        """
        return ((y_pred_m & y_true_m) * weights_m).sum(axis=axis) / weights.sum(axis=axis)

    stratified_cdr = {}
    for strat_name, mask in stratifications.items():
        mask = _repeat_or_flatten(mask, n_repeats)
        cdr_value = _compute_cdr(y_pred[mask], y_true[mask], weights[mask])

        if n_resamples == 0:
            cdr_with_ci = ValueWithCI(value=cdr_value, ci_low=None, ci_upp=None)
        else:
            y_true_resampled, y_pred_sampled, weights_resampled = generate_resamples(
                y_true[mask], y_pred[mask], weights[mask], n_resamples
            )
            cdr_resampled = _compute_cdr(y_true_resampled, y_pred_sampled, weights_resampled, axis=1)
            cdr_with_ci = ValueWithCI(
                value=cdr_value,
                ci_low=np.quantile(cdr_resampled, alpha / 2.0),
                ci_upp=np.quantile(cdr_resampled, 1 - alpha / 2.0),
            )

        stratified_cdr[strat_name] = cdr_with_ci

    return stratified_cdr


def permutation_test(
    y_true: BooleanArray,
    y_treatment: BooleanArray,
    y_control: BooleanArray,
    weights: np.ndarray,
    metric_fns: dict[str, MetricFunc],
    n_permutations: int = 10000,
) -> dict[str, PermTestResult]:
    """
    Computes a p-value by comparing the observed difference in a metric with the randomization distribution.

    Args:
        y_true: (n_samples,) boolean ground-truth labels
        y_treatment: (n_samples, n_preds)
            Boolean predictions for each study and treatment prediction, potentially multiple predictions per
            sample as separate columns.
        y_control: (n_samples, n_preds)
            Reference boolean predictions for each study, potentially multiple predictions per sample as
            separate columns.
        weights: (n_samples,) Per-sample (float) weights from inverse probability weighting
        metric_fns:
            Each keyed callable should follow the `MetricFunc` annotation (see above).
            CAUTION: passing in multiple functions may require to correct for the multiple comparisons problem
                (https://en.wikipedia.org/wiki/Multiple_comparisons_problem), though the correction may happen
                externally and this function is still helpful in avoiding redundant resampling.
        n_permutations:
            The number of samples from which to construct the empirical randomization distribution from. For every
            permutation, the same prediction (per study) is pooled from y_treatment and y_control.

    Returns:
        {"metric_name": PermutationTestResult(delta, p_value), ...}:
            for each metric_fn the observed difference as well as the two-sided p-value

    References:
        Foundations in chapter 15, and optionally 16 of the book:
            Efron, Tibshirani (1994): An Introduction to the Bootstrap
        An application for our use case (compare AI system vs. reader performances on paired data) is given in:
            Mckinney, S. M. et al. (2020). International evaluation of an AI system for breast cancer screening.
                Nature, 577(January). https://doi.org/10.1038/s41586-019-1799-6
    """

    # We have predictions from two groups to compare: y_treatment vs. y_control. NB: do not change this value
    # without carefully adapting the code in this function.
    n_groups = 2

    # The following code requires 2D arrays, => let's make sure we have them.
    if y_treatment.ndim == 1:
        y_treatment = y_treatment[..., None]
    if y_control.ndim == 1:
        y_control = y_control[..., None]

    n_samples, n_preds = y_control.shape
    assert (
        y_treatment.shape == y_control.shape
    ), f"Expecting both y_treatment {y_treatment.shape} and y_control {y_control.shape} to have the same shape."

    # 1. Generate sampling matrix to select from treatment and control predictions (n_permutations, n_samples, n_preds)
    choose_pred_samples = one_hot(
        tensor=np.random.randint(low=0, high=n_preds, size=(n_permutations, n_samples)),
        num_classes=n_preds,
        dtype=y_true.dtype,
    )

    # 2. Generate a 1-hot `choose_group_samples` array for the permutations (n_permutations, n_samples, n_groups)
    group_assignment_indices = np.random.randint(low=0, high=n_groups, size=(n_permutations, n_samples))
    choose_group_samples = one_hot(tensor=group_assignment_indices, num_classes=n_groups, dtype=y_true.dtype)

    def _get_permutations(treatment: np.ndarray, control: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs permutation (random reassignment to one of the two groups) for each of the treatment and control,
        given the two observed arrays. Group 0 is the treatment group, group 1 is the control. For the control
        permutation we simply pool the same values as the treatment but select the other group.
        """
        grouped = np.stack([treatment, control], axis=-1)  # (n_samples, n_preds, n_groups)
        treatment_permuted = np.array(
            [
                grouped[choose_pred][choose_group]
                for choose_pred, choose_group in zip(choose_pred_samples, choose_group_samples)
            ]
        )  # (n_permutations, n_samples)
        control_permuted = np.array(
            [
                grouped[choose_pred][~choose_group]
                for choose_pred, choose_group in zip(choose_pred_samples, choose_group_samples)
            ]
        )  # (n_permutations, n_samples)
        return treatment_permuted, control_permuted

    # 3. Perform prediction pooling & permutation
    y_treatment_permuted, y_control_permuted = _get_permutations(y_treatment, y_control)

    # 4. Permute the labels & weights the same for every permutation. We precompute for reuse by vectorized metric_fns
    y_true_permuted = np.tile(y_true, (n_permutations, 1))  # (n_permutations, n_samples)
    weights_permuted = np.tile(weights, (n_permutations, 1))  # (n_permutations, n_samples)

    #  5. Apply metric_fns in a vectorised fashion
    delta_randomized = {}
    for metric_name, metric_fn in metric_fns.items():
        delta_randomized[metric_name] = metric_fn(
            y_true_permuted, y_treatment_permuted, weights_permuted, axis=1
        ) - metric_fn(y_true_permuted, y_control_permuted, weights_permuted, axis=1)

    # For the observed delta we need to repeat the samples for every `pred` so that we can compute the average.
    # All the arrays created below are (n_samples * n_preds, )
    y_true_repeated = np.repeat(y_true, n_preds)
    weights_repeated = np.repeat(weights, n_preds)

    delta_observed = {
        metric_name: metric_fn(y_true_repeated, y_treatment.flatten(), weights_repeated)
        - metric_fn(y_true_repeated, y_control.flatten(), weights_repeated)
        for metric_name, metric_fn in metric_fns.items()
    }

    p_values = {k: sum(abs(delta_randomized[k]) >= abs(delta_observed[k])) / n_permutations for k in delta_observed}

    return {k: PermTestResult(delta=delta_observed[k], p_value=p_values[k]) for k in delta_observed}


@dataclasses.dataclass
class BootstrapBound:
    value: float
    index: np.ndarray


def generate_bootstrap_samples(
    data: list,
    func: Callable[..., float],
    n_resamples: int,
) -> tuple[list[float], np.ndarray]:
    """
    Compute confidence interval for values of passed function

    Args:
        data: the arguments to func
        func: the function for which to compute the bootstrapped statistics (must return a float).
        n_resamples: the number of bootstrap samples to draw.

    Returns:
        Tuple containing:
            * A list of bootstrap sample results, i.e. the return value of `func` for each bootstrap sample
            * The indices representing the different bootstrap samples
    """

    assert isinstance(data, list)
    n_samples = len(data[0])
    idx = np.random.randint(0, n_samples, (n_resamples, n_samples))

    def select(data, sample):
        return [d[sample] for d in data]

    def evaluate(sample):
        return func(*select(data, sample))

    return list(map(evaluate, idx)), idx


def bootstrap(
    data: list,
    fun: Callable,
    n_resamples: int = 10000,
    alpha: float = 0.05,
) -> tuple[BootstrapBound, BootstrapBound]:

    values, idx = generate_bootstrap_samples(data, fun, n_resamples)

    idx = idx[np.argsort(values, axis=0, kind="mergesort")]
    values = np.sort(values, axis=0, kind="mergesort")

    low = BootstrapBound(value=values[int((alpha / 2.0) * n_resamples)], index=idx[int((alpha / 2.0) * n_resamples)])
    high = BootstrapBound(
        value=values[int((1 - alpha / 2.0) * n_resamples)], index=idx[int((1 - alpha / 2.0) * n_resamples)]
    )

    return low, high


def rates_curve_compute(
    y_true: BooleanArray,
    y_score: np.ndarray,
    n_bootstrap: int,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Compute something like the receiver operating characteristic (ROC) curve but for CDR and recall rate
    rather than TPR and FPR.

    Args:
        y_true (n_samples,), ground-truth binary labels
        y_score (n_samples,), probabilistic score predictions of the positive class
        n_bootstrap: the number of bootstrap samples for the confidence intervals.
        weights (n_samples,), optional set of per-sample (float) weights

    Returns:
        A tuple of equal-sized arrays - recall, cdr, cdr_low, & cdr_high - for a bunch of different thresholds
        applied to y_score. These arrays can be plotted using the `plot_roc_curve` function.
    """

    # Validate input arguments
    assert y_true.shape == weights.shape
    assert y_score.ndim == 1, "y_score should be of shape (n_samples,)"
    assert len(y_true) == len(y_score), "y_true and y_score must both be n_samples long"
    if (num_classes := len(set(y_true))) != 2:
        raise ValueError(f"Need exactly two classes to compute ROC, found {num_classes} in y_true.")

    # Compute recall & CDR over all thresholds (which we get by hijacking sklearn's roc_curve function)
    recall, cdr = [], []
    n_total = weights.sum()
    fpr, _, thresholds = roc_curve(y_true, y_score, pos_label=True, sample_weight=weights)
    for threshold in thresholds:
        y_pred = np.array(y_score >= threshold)
        n_pos = (y_pred * weights).sum()
        n_true_pos = ((y_true & y_pred) * weights).sum()
        recall.append(100 * n_pos / n_total)
        cdr.append(1000 * n_true_pos / n_total)
    recall, cdr = np.array(recall), np.array(cdr)

    if n_bootstrap > 0:

        # We need to define a function here, as the weights have to be part of the bootstrapping (see below).
        # Arguments prefixed with underscore to not shadow variables from outer scope.
        def roc_auc_score_func(_y_true, _y_score, _weights):
            fpr, tpr, _ = roc_curve(_y_true, _y_score, pos_label=True, sample_weight=_weights)
            return auc(fpr, tpr)

        # We bootstrap high and low bounds, and then use the indices with sklearn's ROC function to get the TPR bounds,
        # then interpolate the arrays to the correct length.
        low, high = bootstrap([y_true, y_score, weights], roc_auc_score_func, n_resamples=n_bootstrap, alpha=0.05)
        fpr_low, tpr_low, _ = roc_curve(
            y_true[low.index], y_score[low.index], pos_label=True, sample_weight=weights[low.index]
        )
        fpr_high, tpr_high, _ = roc_curve(
            y_true[high.index], y_score[high.index], pos_label=True, sample_weight=weights[high.index]
        )
        interpolate_low_tpr = interpolate.interp1d(fpr_low, tpr_low, kind="nearest")
        interpolate_high_tpr = interpolate.interp1d(fpr_high, tpr_high, kind="nearest")
        tpr_low, tpr_high = interpolate_low_tpr(fpr), interpolate_high_tpr(fpr)

        # ... before converting those to CDR bounds using a simple miultiplier (i.e. relying on the fact that TPR &
        # CDR only differ in their denominator).
        cdr_mult = (y_true * weights).sum() / weights.sum()
        cdr_low, cdr_high = 1000 * tpr_low * cdr_mult, 1000 * tpr_high * cdr_mult

        return recall, cdr, cdr_low, cdr_high

    return recall, cdr, None, None
