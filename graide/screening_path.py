from typing import Protocol

import numpy as np

from graide import constants as gc
from graide.util import screening_util
from graide.util.types import BooleanArray, Bucket, Decisions, PredsAndMetadata, Workload

RAD, SEND2CC, PROGRAM = gc.DecisionLevel.RAD, gc.DecisionLevel.SEND2CC, gc.DecisionLevel.PROGRAM


class ScreeningPath(Protocol):
    """
    A function that maps a given bucket (mask of studies) at one decision level to boolean decisions and integer
    workload arrays at a (possibly different) decision level. The Decisions and Workload arrays returned should be
    the same shape as each other, either (n_studies_in_bucket,) or (n_studies_in_bucket, 2).
    """

    def __call__(
        self,
        bucket: Bucket,
        preds_and_md: PredsAndMetadata,
        input_decision_level: gc.DecisionLevel,
        output_decision_level: gc.DecisionLevel,
        screening_system: gc.ScreeningSystem = gc.ScreeningSystem.GERMANY,
    ) -> tuple[Decisions, Workload]:
        ...


def direct_no_recall(
    bucket: Bucket,
    preds_and_md: PredsAndMetadata,
    input_decision_level: gc.DecisionLevel,
    output_decision_level: gc.DecisionLevel,
    screening_system: gc.ScreeningSystem = gc.ScreeningSystem.GERMANY,
) -> tuple[Decisions, Workload]:
    """This bucket represents a "no recall" decision made directly, without the screening system."""

    assert input_decision_level in {SEND2CC, PROGRAM}
    n_bucket = bucket.sum()
    return np.zeros(n_bucket, dtype=bool), np.zeros(n_bucket, dtype=int)


def direct_recall(
    bucket: Bucket,
    preds_and_md: PredsAndMetadata,
    input_decision_level: gc.DecisionLevel,
    output_decision_level: gc.DecisionLevel,
    screening_system: gc.ScreeningSystem = gc.ScreeningSystem.GERMANY,
) -> tuple[Decisions, Workload]:
    """
    This bucket represents a recall decision made directly, without the screening system. The bucket has no need
    for converting between decision levels as it only makes sense for `input_decision_level=PROGRAM`.
    """

    assert input_decision_level is PROGRAM
    n_bucket = bucket.sum()
    return np.ones(n_bucket, dtype=bool), np.zeros(n_bucket, dtype=int)


def negative_ai_read(
    bucket: Bucket,
    preds_and_md: PredsAndMetadata,
    input_decision_level: gc.DecisionLevel,
    output_decision_level: gc.DecisionLevel,
    screening_system: gc.ScreeningSystem = gc.ScreeningSystem.GERMANY,
) -> tuple[Decisions, Workload]:
    """
    This bucket represents a negative rad-level decision, => requiring checking the other radiologist in order to
    determine the decisions and workload arrays. We need to return arrays with a decision dimension here
    (i.e. (n_studies_in_bucket, 2)) in order to simulate the impact of making these decisions for each radiologist
    separately.
    """

    assert input_decision_level is gc.DecisionLevel.RAD

    n_bucket = bucket.sum()

    # We start with negative predictions for all studies in the bucket.
    decisions = np.zeros(n_bucket, dtype=bool)

    # We start with 0 workload here because we're making a negative rad-level decision without using the radiologist.
    workload = np.zeros(n_bucket, dtype=int)

    # If we're evaluating the German screening system then we create arrays with 2 decisions / workload values per
    # study in order to simulate the impact of making this decision for each radiologist separately. If the output
    # decision level is send2cc or beyond, these decisions will be paired with each "other" reader within the
    # `rad_to_send2cc` function.
    if screening_system is gc.ScreeningSystem.GERMANY:
        decisions = np.c_[decisions, decisions]
        workload = np.c_[workload, workload]

    if output_decision_level is RAD:
        return decisions, workload

    decisions, workload, disagreement = rad_to_send2cc(
        bucket, decisions, workload, preds_and_md.y_rad, screening_system
    )
    if output_decision_level is SEND2CC:
        return decisions, workload

    return send2cc_to_program(
        bucket,
        decisions,
        workload,
        preds_and_md.y_send2cc,
        preds_and_md.y_program,
        disagreement,
        screening_system=screening_system,
    )


def screening_system_1_reader(
    bucket: Bucket,
    preds_and_md: PredsAndMetadata,
    input_decision_level: gc.DecisionLevel,
    output_decision_level: gc.DecisionLevel,
    screening_system: gc.ScreeningSystem = gc.ScreeningSystem.GERMANY,
) -> tuple[Decisions, Workload]:
    """
    This bucket represents a deferral that is sent to a single radiologist for a _final_ decision. We need to return
    arrays with a decision dimension here (i.e. (n_studies_in_bucket, 2)) in order to simulate the impact of making
    these decisions for each radiologist separately.
    """

    assert input_decision_level is gc.DecisionLevel.RAD

    # We start with the reader-level decisions
    decisions = preds_and_md.y_rad[bucket]

    # If evaluating the UK screening system we only return that of the first reader as the second isn't independent.
    if screening_system is gc.ScreeningSystem.UK:
        decisions = decisions[:, 0]

    # We have one unit of work per decision.
    workload = np.ones_like(decisions, dtype=int)

    return decisions, workload


def screening_system_2_readers(
    bucket: Bucket,
    preds_and_md: PredsAndMetadata,
    input_decision_level: gc.DecisionLevel,
    output_decision_level: gc.DecisionLevel,
    screening_system: gc.ScreeningSystem = gc.ScreeningSystem.GERMANY,
) -> tuple[Decisions, Workload]:
    """
    This bucket represents the complete deferral to the program's decision at any decision level. In other words,
    this bucket returns the decisions made by the program at the intended output decision-level and the full
    workload done by the program to make them.

    The shape of the output arrays is determined by the input decision level, namely by adding a decision dimension
    only if the input decision-level is RAD.
    """

    # We simply take the reference decisions at the output decision level and initialise the workload as a single unit
    # of work per study.
    decisions = screening_util.get_reference_decisions(preds_and_md, output_decision_level)[bucket]
    workload = np.ones(bucket.sum(), dtype=int)

    # If we're evaluating the Germans screening system with a strategy that operates on the rad-level then we need to
    # add a decision dimension to our workload array to represent one unit of work per radiologist. This allows us to
    # simulate our pairing with each radiologist separately.
    if input_decision_level is RAD and screening_system is gc.ScreeningSystem.GERMANY:
        workload = np.c_[workload, workload]

    # If we're evaluating the UK screening system on a rad-level then we only want to take the first reader's decision
    # as that's the only independent read.
    if output_decision_level is RAD and screening_system is gc.ScreeningSystem.UK:
        decisions = decisions[:, 0]

    # If we're evaluating this bucket on the send2cc-level or program-level then we need to add another read to
    # represent the work done by the 'other' reader ...
    if output_decision_level in {SEND2CC, PROGRAM}:
        workload += 1

    # ... & for program-level evaluation we need to account for those studies that went to CC.
    if output_decision_level is PROGRAM:
        workload[preds_and_md.y_send2cc[bucket]] += 1

    return decisions, workload


def suspicious_ai_read(
    bucket: Bucket,
    preds_and_md: PredsAndMetadata,
    input_decision_level: gc.DecisionLevel,
    output_decision_level: gc.DecisionLevel,
    screening_system: gc.ScreeningSystem = gc.ScreeningSystem.GERMANY,
) -> tuple[Decisions, Workload]:
    """
    This bucket represents a suspicious rad-level decision made by the AI. Even though this will obviously lead to a
    positive send2cc-level decision, we still need to consider the other radiologist when evaluating workload. We need
    to return arrays with a decision dimension here (i.e. (n_studies_in_bucket, 2)) in order to simulate the impact
    of making these decisions for each radiologist separately.
    """

    assert input_decision_level is RAD

    n_bucket = bucket.sum()

    # We start with positive predictions for all studies in the bucket.
    decisions = np.ones(n_bucket, dtype=bool)

    # We start with 0 workload here because we're making a positive rad-level decision without using the radiologist.
    workload = np.zeros(n_bucket, dtype=int)

    # If we're evaluating the German screening system then we create arrays with 2 decisions / workload values per
    # study in order to simulate the impact of making this decision for each radiologist separately. If the output
    # decision level is send2cc or beyond, these decisions will be paired with each "other" reader within the
    # `rad_to_send2cc` function.
    if screening_system is gc.ScreeningSystem.GERMANY:
        decisions = np.c_[decisions, decisions]
        workload = np.c_[workload, workload]

    if output_decision_level is RAD:
        return decisions, workload

    decisions, workload, disagreement = rad_to_send2cc(
        bucket, decisions, workload, preds_and_md.y_rad, screening_system
    )
    if output_decision_level is SEND2CC:
        return decisions, workload

    return send2cc_to_program(
        bucket,
        decisions,
        workload,
        preds_and_md.y_send2cc,
        preds_and_md.y_program,
        disagreement,
        screening_system=screening_system,
    )


def rad_to_send2cc(
    bucket: Bucket,
    decisions: Decisions,
    workload: Workload,
    reference_decisions_rad: Decisions,
    screening_system: gc.ScreeningSystem = gc.ScreeningSystem.GERMANY,
) -> tuple[Decisions, Workload, BooleanArray]:
    """
    Converts rad-level decisions and workload to send2cc-level.

    Returns:
        The decisions and workload, converted to send2cc-level, and a mask indicating the studies where the AI and
        the "other" radiologist disagreed. This is only needed when working with the UK screening system where the
        subsequent step (`send2cc_to_program`) depends on whether the send2cc decision was unanimous or not.
    """

    # Ensure our dimensions are correct
    if screening_system is gc.ScreeningSystem.UK:
        assert (decisions.shape == workload.shape) and (decisions.ndim == workload.ndim == 1), (
            "Both input arrays must be of shape (n_studies_in_bucket, ). Yours were: ",
            f"decisions={decisions.shape}, workload={workload.shape}",
        )

    else:
        assert (decisions.shape == workload.shape) and (decisions.shape[1] == reference_decisions_rad.shape[1]), (
            "Both input arrays must be of shape (n_studies_in_bucket, 2). Yours were: "
            f"decisions={decisions.shape}, workload={workload.shape}"
        )

    # If we're evaluating the UK screening system then we know that the bucket that called this function only
    # replaced the 2nd (non-independent) reader, => to make our disagreement consideration here we consider
    # the first reader.
    if screening_system is gc.ScreeningSystem.UK:
        reference_decisions_rad = reference_decisions_rad[:, 0]

    # We define a mask indicating the studies in this bucket where the AI and the reference disagree, used in the UK
    # screening system to determine program-level decisions.
    disagreement = decisions != reference_decisions_rad[bucket]

    if screening_system is gc.ScreeningSystem.GERMANY:

        # To convert to a send2cc-level decision we need to pair each rad-level decision with the other reader,
        # => giving us two versions of send2cc. By "pair" here we mean consider what the send2cc decision would be via
        # applying a logical OR over the AI read coming from some bucket and the other reader's read. The resulting
        # decisions array is of shape (n_studies_in_bucket, 2).
        decisions = np.c_[
            decisions[:, 0] | reference_decisions_rad[bucket, 1],
            decisions[:, 1] | reference_decisions_rad[bucket, 0],
        ]

    elif screening_system is gc.ScreeningSystem.UK:

        # Any decisions where the AI & reference agree are final (=> the AI decision is unchanged). All others are
        # positive because at least one of the two reads must have been positive.
        decisions[disagreement] = 1

    # Here we add one unit of work to each decision to represent the work done by the "other" reader in converting from
    # a rad-level to a send2cc-level decision (i.e. the work done in the above conditional clauses).
    workload += 1

    # If evaluating in the UK, we 'disregard' the AI decision if the AI disagrees with the first reader. Put
    # differently, in case of disagreement we use the original readers to determine whether 1) we need an arbitration
    # read, and 2) if not, what the final decision should be. In both cases we need to add a unit of work to reflect
    # the fact that we used the second reader.
    if screening_system is gc.ScreeningSystem.UK:
        workload[disagreement] += 1

    return decisions, workload, disagreement


def send2cc_to_program(
    bucket: Bucket,
    decisions: Decisions,
    workload: Workload,
    reference_decisions_send2cc: Decisions,
    reference_decisions_program: Decisions,
    disagreement: BooleanArray,
    screening_system: gc.ScreeningSystem = gc.ScreeningSystem.GERMANY,
):
    """
    Converts send2cc-level decisions and workload to program-level.

    Args:
        disagreement:
            A boolean mask indicating the studies in this bucket for which the two 'readers' disagreed (where readers
            is in quotes because one 'reader' could be an AI-based strategy).
    """

    # We may need to duplicate our reference decisions to create (n_studies_in_bucket, 2)-shaped arrays
    if screening_util.has_decision_dim(decisions):
        reference_decisions_send2cc = np.c_[reference_decisions_send2cc, reference_decisions_send2cc]
        reference_decisions_program = np.c_[reference_decisions_program, reference_decisions_program]

    # In the UK screening system, we only consider positive decisions as positive _send2cc_ decisions if there was
    # disagreement between the two reads. All other decisions, either positive or negative, are taken as final. For the
    # positive send2cc decisions, we either have a historical arbitration (~ CC) decision, or we don't. If we do then
    # we use it, but if we don't then we want to stick with the program's historical negative decision anyway. We can
    # implement these two cases very simple as: if there is disagreement, take the program's historical decision.
    if screening_system is gc.ScreeningSystem.UK:
        positive_send2cc_decisions = decisions & disagreement
        decisions[positive_send2cc_decisions] = reference_decisions_program[bucket][positive_send2cc_decisions]

        # The following mask indicates the subset of the `positive_send2cc_decisions` where an arbitration read was
        # used, necessary for the workload calculation below.
        use_cc_decision = positive_send2cc_decisions & reference_decisions_send2cc[bucket]

    # In the German screening system, we want to use the decision made by the CC (i.e. the program-level reference
    # decision) for those studies where the send2cc-level decision is positive and we have a retrospective CC. For all
    # other studies, positive or negative, we leave the decision as-is in order to put the onus on the AI.
    else:
        use_cc_decision = decisions & reference_decisions_send2cc[bucket]
        decisions[use_cc_decision] = reference_decisions_program[bucket][use_cc_decision]

    # We add one unit of work for all studies where the CC / arbitration was used.
    workload[use_cc_decision] += 1

    return decisions, workload
