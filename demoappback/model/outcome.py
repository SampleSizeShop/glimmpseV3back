from demoappback.model.isu_factor import IsuFactor, IsuFactorType, IsuFactorNature, IsuFactorHypothesisType


class Outcome(IsuFactor):
    """
    Class describing outcomes.
    """

    def __init__(self, name: str = None, standard_deviation: float = 1):
        super().__init__(name, IsuFactorNature.WITHIN, IsuFactorType.OUTCOME, IsuFactorHypothesisType.GLOBAL_TRENDS)
        self.standard_deviation = standard_deviation
