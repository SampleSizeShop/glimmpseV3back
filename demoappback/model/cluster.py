from demoappback.model.isu_factor import IsuFactor, IsuFactorType, IsuFactorNature


class Cluster(IsuFactor):
    """
    Class describing outcomes.
    """

    def __init__(self, name: str = None, standard_deviation = 1):
        super.__init__(self, name, IsuFactorNature.WITHIN, IsuFactorType.CLUSTER)
        self.standard_deviation = standard_deviation
