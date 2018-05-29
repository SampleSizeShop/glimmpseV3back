from enum import Enum


class HypothesisType(Enum):
    GLOBAL_TRENDS = 1
    IDENTITY = 2
    POLYNOMIAL = 3
    USER_DEFINED = 4


class IsuFactorType(Enum):
    OUTCOME = 1
    REPEATED_MEASURE = 2
    CLUSTER = 3
    PREDICTOR = 4


class Nature(Enum):
    WITHIN = 1
    BETWEEN = 2


class TargetEvent(Enum):
    REJECTION = 1
    CI_WIDTH = 2
    WAVR = 3


class SolveFor(Enum):
    POWER = 1
    SAMPLESIZE = 2

class ClType(Enum):
    CLTYPE_DESIRED = 1
    CLTYPE_NOT_DESIRED = 2