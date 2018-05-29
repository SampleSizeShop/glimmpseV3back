from demoappback.model.isu_factor import IsuFactor, IsuFactorType, Nature, HypothesisType


class Predictor(IsuFactor):
    """
    Class describing between predictors.
    """

    def __init__(self,
                 name: str=None,
                 values: []=None,
                 in_hypothesis: bool=False,
                 child=None,
                 partial_matrix=None):
        super().__init__(name=name,
                         nature=Nature.BETWEEN,
                         factor_type=IsuFactorType.PREDICTOR,
                         values=values,
                         in_hypothesis=in_hypothesis,
                         hypothesis_type=HypothesisType.GLOBAL_TRENDS,
                         child=child,
                         partial_matrix=partial_matrix)
