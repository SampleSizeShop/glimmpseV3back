from enum import Enum

class IsuFactor(object):
    """
    Parent class for indepenent sampling unit factors (ISUs).

    Outcomes
    Repeated Measures
    Clusters
    """


    def __init__(self, name: str = None, nature: str = None, isu_factor_type: IsuFactorType = None):
        self.name = name
        self.nature = nature
        self.isu_factor_type = isu_factor_type
        self.nature = None
        self.values = []
        self.child = None
        self.partialMatrix = None
        self.inHypothesis = False
        self.isuFactorNature = IsuFactorNature.GLOBAL_TRENDS


class IsuFactorNature(Enum):
    GLOBAL_TRENDS = 1
    IDENTITY = 2
    POLYNOMIAL = 3
    USER_DEFINED = 4


class IsuFactorType(Enum):
    OUTCOME = 1
    REPEATED_MEASURE = 2
    CLUSTER = 3


class IsuFactorNature(Enum):
    WITHIN = 1
    BETWEEN = 2
