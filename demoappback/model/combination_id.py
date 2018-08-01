class CombinationId(object):
    """
    Id Class for combinations of ISU Factors.
    """

    def __init__(self,
                 factorName: str = None,
                 factorType: str = None,
                 value: [] = None,
                 order: int = 0,
                 **kwargs):
        self.factorName = factorName
        self.factorType = factorType
        self.value = value

        if kwargs.get('source'):
            self.from_dict(kwargs['source'])
