from demoappback.model.enums import HypothesisType, IsuFactorType


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
                 hypothesis_type: HypothesisType =None,
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
