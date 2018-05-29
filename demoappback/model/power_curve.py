from demoappback.model.isu_factor import IsuFactor
from demoappback.model.enums import HypothesisType, IsuFactorType, Nature

class ConfidenceInterval(object):
    """
    Class describing a power curve confidence interval
    """
    def __init__(self,
                 assumptions = None,
                 lower_tail_probability: float=0,
                 upper_tail_probability: float=1,
                 beta_sample_size: int=2,
                 beta_sigma_rank: float=1):
        self.assumptions = assumptions
        self.lowerTailProbability = lower_tail_probability
        self.upperTailProbability = upper_tail_probability
        self.betaSamplesize = beta_sample_size
        self.betasigmaRank = beta_sigma_rank

class DataSeries(object):
    """
    Class describing a power curve data series
    """

    def __init__(self,
                 type_I_error: float=0.05,
                 mean_scale_factor: float=1,
                 variance_scale_factor: float=3):
        self.type_I_error = type_I_error
        self.mean_scale_factor = mean_scale_factor
        self.variance_scale_factor = variance_scale_factor


class PowerCurve(object):
    """
    Class describing outcomes.
    """

    def __init__(self,
                 confidence_interval: ConfidenceInterval=None,
                 x_axis: str=None,
                 data_series: []=None):
        self.confidence_interval = confidence_interval
        self.x_axis = x_axis
        self.data_series = data_series
