from enum import Enum


class HypothesisType(Enum):
    GLOBAL_TRENDS = 'Global Trends'
    IDENTITY = 'Identity'
    POLYNOMIAL = 'Polynomial'
    USER_DEFINED = 'Define partial contrasts'
    CUSTOM_C_MATRIX = 'Define hypothesis C Matrix'
    CUSTOM_U_MATRIX = 'Define hypothesis U Matrix'


class IsuFactorType(Enum):
    OUTCOME = 'Outcome'
    REPEATED_MEASURE = 'Repeated Measure'
    CLUSTER = 'Cluster'
    PREDICTOR = 'Between ISU Predictor'


class Nature(Enum):
    WITHIN = 'Within'
    BETWEEN = 'Between'


class TargetEvent(Enum):
    REJECTION = 'REJECTION'
    CI_WIDTH = 'CI_WIDTH'
    WAVR = 'WAVR'


class SolveFor(Enum):
    POWER = 'POWER'
    SAMPLESIZE = 'SAMPLESIZE'

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


class PolynomialMatrices(Enum):
    LINEAR_POLYNOMIAL_CMATRIX = [-1, 1]
    QUADRATIC_POLYNOMIAL_CMATRIX = [[-1, 0, 1],
                                   [1, -2, 1]]
    CUBIC_POLYNOMIAL_CMATRIX = [[-3, -1, 1, 3],
                                [1, -1, -1, 1],
                                [-1, 3, -3, 1]]
    QUINTIC_POLYNOMIAL_CMATRIX = [[-2, -1, 0, 1, 2],
                                  [2, -1, -2, -1, 2],
                                  [-1, 2, 0, -2, 1],
                                  [1, -4, 6, -4, 1]]
    SEXTIC_POLYNOMIAL_CMATRIX = [[-5, -3, -1, 1, 3, 5],
                                 [5, -1, -4, -4, -1, 5],
                                 [-5, 7, 4, -4, -7, 5],
                                 [1, -3, 2, 2, -3, 1],
                                 [-1, 5, -10, 10, -5, 1]]

class OptionalArgs(Enum):
    APPROXIMATION = 'approximation'
    EPSILON_ESTIMATOR = 'epsilon_estimator'
    UNIREPMETHOD = 'unirepmethod'
    N_EST = 'n_est'
    RANK_EST = 'rank_est'
    ALPHA_CL = 'alpha_cl'
    ALPHA_CU = 'alpha_cu'
    N_IP = 'n_ip'
    RANK_IP = 'rank_ip'
    TOLERANCE = 'tolerance'
