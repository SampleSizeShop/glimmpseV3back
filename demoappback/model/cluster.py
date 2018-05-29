from demoappback.model.isu_factor import IsuFactor, IsuFactorType, Nature, HypothesisType


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
                 cluster_level: int=0):
        super().__init__(name=name,
                         nature=Nature.WITHIN,
                         factor_type=IsuFactorType.CLUSTER,
                         values=values,
                         in_hypothesis=in_hypothesis,
                         hypothesis_type=HypothesisType.GLOBAL_TRENDS,
                         child=child,
                         partial_matrix=partial_matrix)
        self.cluster_level = cluster_level
