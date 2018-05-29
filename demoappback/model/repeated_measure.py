import numpy as np

from demoappback.model.isu_factor import IsuFactor, IsuFactorType, Nature, HypothesisType


class RepeatedMeasure(IsuFactor):
    """
    Class describing repeated measures.
    """

    def __init__(self,
                 name: str=None,
                 values: []=None,
                 in_hypothesis: bool=False,
                 child=None,
                 partial_matrix=None,
                 units: str=None,
                 type: str=None,
                 no_repeats: int=2,
                 partial_u_matrix: []=[],
                 correlation_matrix: []=[]):
        super().__init__(name=name,
                         nature=Nature.WITHIN,
                         factor_type=IsuFactorType.REPEATED_MEASURE,
                         values=values,
                         in_hypothesis=in_hypothesis,
                         hypothesis_type=HypothesisType.GLOBAL_TRENDS,
                         child=child,
                         partial_matrix=partial_matrix)
        self.units = units
        self.type = type
        self.no_repeats = no_repeats
        self.partial_u_matrix = np.matrix(partial_u_matrix)
        self.correlation_matrix = np.matrix(correlation_matrix)
