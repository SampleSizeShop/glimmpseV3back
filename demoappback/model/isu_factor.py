from enum import Enum
import numpy as np

class IsuFactor(object):
    """
    Parent class for indepenent sampling unit factors (ISUs).

    Outcomes
    Repeated Measures
    Clusters
    """

    def __init__(self,
                 name: str=None,
                 nature: str=None,
                 factor_type: IsuFactorType=None,
                 values: []=None, in_hypothesis: bool=False,
                 hypothesis_type: HypothesisType=None,
                 child=None,
                 partial_matrix=None):
        self.name = name
        self.nature = nature
        self.factor_type = factor_type
        self.values = values
        self.in_hypothesis = in_hypothesis
        self.hypothesis_type = hypothesis_type
        self.child = child
        self.partialMatrix = partial_matrix


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

