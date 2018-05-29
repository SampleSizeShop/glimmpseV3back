from demoappback.model.isu_factor import IsuFactor, IsuFactorType, Nature, HypothesisType


class Outcome(IsuFactor):
    """
    Class describing outcomes.
    """

    def __init__(self,
                 name: str = None,
                 values: [] = None,
                 in_hypothesis: bool = False,
                 hypothesis_type: HypothesisType = HypothesisType.GLOBAL_TRENDS,
                 child = None,
                 partial_matrix = None,
                 standard_deviation: float = 1):
        super().__init__(name=name,
                         nature=Nature.WITHIN,
                         factor_type=IsuFactorType.OUTCOME,
                         values=values,
                         in_hypothesis=in_hypothesis,
                         hypothesis_type=HypothesisType.GLOBAL_TRENDS,
                         child=child,
                         partial_matrix=partial_matrix)
        self.standard_deviation = standard_deviation


name: str = None,
nature: str = None,
factor_type: IsuFactorType = None,
values: [] = None,
in_hypothesis: bool = False,
hypothesis_type: HypothesisType = None,
child = None,
partial_matrix = None