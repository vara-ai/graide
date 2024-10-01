import enum

SENS = "sens"
SPEC = "spec"
CDR = "cdr"
RECALL = "recall"
METRIC_NAMES = [SENS, SPEC, CDR, RECALL]

CDR_RESULT_FN = "result_cdr_optimising.pklz"
RECALL_RESULT_FN = "result_recall_optimising.pklz"
ORACLE_RESULT_FN = "result_oracle.pklz"


@enum.unique
class DecisionLevel(enum.Enum):
    RAD = "Rad."

    # acting at the level of sending studies —> consensus (i.e. as if considering a pair of radiologists)
    SEND2CC = "Send2CC"

    # acting at the program i.e. consensus conference level (all studies recalled by consensus)
    PROGRAM = "Program"

    def is_greater_than(self, other_dl: "DecisionLevel") -> bool:
        """Returns whether this decision level (i.e. self) is greater than `other_dl`."""
        if self is DecisionLevel.PROGRAM and other_dl is not DecisionLevel.PROGRAM:
            return True
        elif self is DecisionLevel.SEND2CC and other_dl is DecisionLevel.RAD:
            return True
        return False


@enum.unique
class ScreeningSystem(enum.Enum):

    # The Swedish screening system is the same as the German (at least with respect to the level of detail of this
    # simulation), => please also use GERMANY for all Swedish evaluations.
    GERMANY = "Germany"

    UK = "UK"
