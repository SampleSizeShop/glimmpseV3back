class OutcomeRepeatedMeasureStDev(object):
    """Class to describe outcome repeated measure st deviations"""

    def __init__(self,
                 outcome: str=None,
                 repeated_measure: str=None,
                 values: []=None):
        self.outome = outcome
        self.repeated_measure = repeated_measure
        self.values = values


class IsuFactors(object):
    """
    Parent class for indepenent sampling unit factors (ISUs).

    Outcomes
    Repeated Measures
    Clusters
    """

    def __init__(self,
                 variables: []=None,
                 between_isu_relative_group_sizes: []=None,
                 marginal_means: []=None,
                 smallest_group_size: int=None,
                 outcome_correlation_matrix=None,
                 outcome_repeated_measure_st_devs=None):
        self.variables = variables
        self.between_isu_relative_group_sizes = between_isu_relative_group_sizes
        self.marginal_means = marginal_means
        self.smallest_group_size = smallest_group_size
        self.outcome_correlation_matrix = outcome_correlation_matrix
        self.outcome_repeated_measure_st_devs = outcome_repeated_measure_st_devs
