from demoappback.model.isu_factor import IsuFactor
from demoappback.model.enums import HypothesisType, IsuFactorType, Nature


class ClusterLevel(object):
    """
    Class describing cluster levels
    """

    def __init__(self,
                 level_name: int = None,
                 no_elements: int = 1,
                 intra_class_correlation = 1):
        self.level_name = level_name
        self.no_elements = no_elements
        self.intra_class_corellation = intra_class_correlation


class Cluster(IsuFactor):
    """
    Class describing clusters.
    """

    def __init__(self,
                 name: str=None,
                 values: []=None,
                 in_hypothesis: bool=False,
                 child=None,
                 partial_matrix=None,
                 levels: [] = None):
        super().__init__(name=name,
                         nature=Nature.WITHIN,
                         factor_type=IsuFactorType.CLUSTER,
                         values=values,
                         in_hypothesis=in_hypothesis,
                         hypothesis_type=HypothesisType.GLOBAL_TRENDS,
                         child=child,
                         partial_matrix=partial_matrix)
        self.levels = levels
