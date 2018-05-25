from demoappback.model.isu_factor import IsuFactor, IsuFactorType, IsuFactorNature


class Cluster(IsuFactor):
    """
    Class describing clusters.
    """

    def __init__(self, name: str = None, cluster_level: int = 0):
        super().__init__(name, IsuFactorNature.WITHIN, IsuFactorType.CLUSTER)
        self.cluster_level = cluster_level
