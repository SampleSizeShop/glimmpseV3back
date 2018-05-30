from demoappback.model.isu_factor import IsuFactor
from demoappback.model.enums import HypothesisType, IsuFactorType, Nature


class Outcome(IsuFactor):
    """
    Class describing outcomes.
    """

    def __init__(self,
                 name: str=None,
                 values: []=None,
                 in_hypothesis: bool=False,
                 child=None,
                 partial_matrix=None,
                 standard_deviation: float=1,
                 **kwargs):
        super().__init__(name=name,
                         nature=Nature.WITHIN,
                         factor_type=IsuFactorType.OUTCOME,
                         values=values,
                         in_hypothesis=in_hypothesis,
                         hypothesis_type=HypothesisType.GLOBAL_TRENDS,
                         child=child,
                         partial_matrix=partial_matrix)
        self.standard_deviation = standard_deviation
        if kwargs.get('source'):
            self.from_dict(kwargs['source'])

    def from_dict(self, source):
        super().from_dict(source)
        if source.get('standardDeviation'):
            self.standard_deviation = source['standardDeviation']
