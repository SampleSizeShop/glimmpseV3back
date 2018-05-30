from demoappback.model.isu_factor import IsuFactor
from demoappback.model.enums import HypothesisType, IsuFactorType, Nature


class Predictor(IsuFactor):
    """
    Class describing between predictors.
    """

    def __init__(self,
                 name: str=None,
                 values: []=None,
                 in_hypothesis: bool=False,
                 child=None,
                 partial_matrix=None,
                 **kwargs):
        super().__init__(name=name,
                         nature=Nature.BETWEEN,
                         factor_type=IsuFactorType.PREDICTOR,
                         values=values,
                         in_hypothesis=in_hypothesis,
                         hypothesis_type=HypothesisType.GLOBAL_TRENDS,
                         child=child,
                         partial_matrix=partial_matrix)

        if kwargs.get('source'):
            pass
