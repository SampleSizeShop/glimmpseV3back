class ConfidenceInterval(object):
    """
    Class holding required information to calculate confidence intervals.
    """

    def __init__(self,
                 beta_known: bool = True,
                 lower_tail: float =0.05,
                 upper_tail: float =0.0,
                 rank_est: int =1,
                 n_est:int = 1,
                 **kwargs):
        """
        Class describing a Confidence Interval
        :param beta_known:
        :param lower_tail:
        :param upper_tail:
        :param rank_est:
        :param n_est:
        :param kwargs:
        """
        self.beta_known = beta_known
        self.lower_tail = lower_tail
        self.upper_tail = upper_tail
        self.rank_est = rank_est
        self.n_est = n_est
        if kwargs.get('source'):
            self.from_dict(kwargs['source'])

    def from_dict(self, source):
        for key, value in source.items():
            if key == 'beta_known':
                self.beta_known = value
            if key == 'lower_tail':
                self.lower_tail = value
            if key == 'upper_tail':
                self.upper_tail = value
            if key == 'rank_est':
                self.rank_est = value
            if key == 'n_est':
                self.n_est = value

    def to_dict(self):
        ret = dict(beta_known=self.beta_known,
                   lower_tail=self.lower_tail,
                   upper_tail=self.upper_tail,
                   rank_est=self.rank_est,
                   n_est=self.n_est)
        return ret