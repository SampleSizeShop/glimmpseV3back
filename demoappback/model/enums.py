from enum import Enum


class HypothesisType(Enum):
    GLOBAL_TRENDS = 'GLOBAL_TRENDS'
    IDENTITY = 'IDENTITY'
    POLYNOMIAL = 'POLYNOMIAL'
    USER_DEFINED = 'USER_DEFINED'


class IsuFactorType(Enum):
    OUTCOME = 'Outcome'
    REPEATED_MEASURE = 'Repeated Measure'
    CLUSTER = 'Cluster'
    PREDICTOR = 'Between ISU Predictor'


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


class Tests(Enum):
    HOTELLING_LAWLEY = 'Hotelling Lawley Trace'
    PILLAI_BARTLET = 'Pillai-Bartlett Trace'
    WILKS_LIKLIEHOOD = 'Wilks Likelihood Ratio'
    BOX_CORRECTION = 'Repeated Measures: Box Correction'
    GEISSER_GREENHOUSE = 'Repeated Measures: Geisser-Greenhouse Correction'
    HUYNH_FELDT = 'Repeated Measures: Huynh-Feldt Correction'
    UNCORRECTED = 'Repeated Measure: uncorrected'
    UNIREP = 'High Dimension low sample size- UNIREP'
    MULTIREP = 'High Dimension low sample size - MULTIREP'
