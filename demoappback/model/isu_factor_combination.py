from demoappback.model.combination_id import CombinationId


class IsuFactorCombination(object):
    """
    Id Class for combinations of ISU Factors.
    """

    def __init__(self,
                 id: CombinationId = None,
                 value: float = None,
                 **kwargs):
        self.id = id
        self.value = value

        if kwargs.get('source'):
            self.from_dict(kwargs['source'])
