from app.calculation_service.model.confidence_interval import ConfidenceInterval


class DataSeries(object):
    """
    Class describing a power curve data series
    """

    def __init__(self,
                 type_I_error: float=0.05,
                 mean_scale_factor: float=1,
                 variance_scale_factor: float=3,
                 **kwargs):
        self.type_I_error = type_I_error
        self.mean_scale_factor = mean_scale_factor
        self.variance_scale_factor = variance_scale_factor

        if kwargs.get('source'):
            self.from_dict(kwargs['source'])

    def from_dict(self, source):
        if source.get('_typeIerror'):
            self.type_I_error = source['_typeIerror']
        if source.get('_meanScaleFactor'):
            self.mean_scale_factor = source['_meanScaleFactor']
        if source.get('_varianceScaleFactor'):
            self.variance_scale_factor = source['_varianceScaleFactor']


class PowerCurve(object):
    """
    Class describing outcomes.
    """

    def __init__(self,
                 confidence_interval: ConfidenceInterval=None,
                 x_axis: str=None,
                 data_series: []=None,
                 **kwargs):
        self.confidence_interval = confidence_interval
        self.x_axis = x_axis
        self.data_series = data_series

        if kwargs.get('source'):
            self.from_dict(kwargs['source'])

    def from_dict(self, source):
        if source.get('_confidenceInterval'):
            self.confidence_interval = ConfidenceInterval(source=source['_confidenceInterval'])
        if source.get('_xAxis'):
            self.x_axis = source['_xAxis']
        if source.get('_dataSeries'):
            self.data_series = [DataSeries(source=ds) for ds in source['_dataSeries']]
