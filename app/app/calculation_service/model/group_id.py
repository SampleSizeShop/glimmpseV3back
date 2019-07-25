class GroupId(object):
    """Hashable data object used when colappsing group ids"""

    def __init__(self,
                identifier: [] = None):
        self.identifier = identifier

    def __hash__(self):
        """Compare group ids by name, factorName and vaLue. Id's should be a list of dict
         each dict chould contain order, factorName, factorType and value"""
        key = ''
        for i in self.identifier:
            key = key + 'order' + str(i.get('order')) + 'factorName' + str(i.get('factorName')) + 'value' + str(i.get('value'))
        return hash(key)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()
