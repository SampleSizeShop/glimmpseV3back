from demoappback.model.isu_factor import IsuFactor, IsuFactorType, IsuFactorNature


class Outcome(IsuFactor):
    """
    Class describing outcomes.
    """

    def __init__(self, name: str = None):
        super.__init__(self, name, IsuFactorNature.WITHIN, IsuFactorType.OUTCOME)
        self.standard_deviation = None
