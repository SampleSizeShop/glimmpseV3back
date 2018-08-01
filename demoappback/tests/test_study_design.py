import unittest
import numpy as np
from pyglimmpse import unirep


from demoappback.model.cluster import Cluster, ClusterLevel
from demoappback.model.enums import TargetEvent, SolveFor, Tests
from demoappback.model.isu_factors import IsuFactors, OutcomeRepeatedMeasureStDev
from demoappback.model.linear_model import LinearModel
from demoappback.model.outcome import Outcome
from demoappback.model.power_curve import PowerCurve, ConfidenceInterval, DataSeries
from demoappback.model.predictor import Predictor
from demoappback.model.repeated_measure import RepeatedMeasure
from demoappback.model.study_design import StudyDesign
from demoappback.models import Matrix


class StudyDesignTestCase(unittest.TestCase):
    m = Matrix('M')

    def setUp(self):
        self.m = Matrix('M')

    def tearDown(self):
        pass

    def test___init__(self):
        """Should return a matrix with name M and vaules _SAMPLE"""
        expected = StudyDesign(
                isu_factors=None,
                target_event=None,
                solve_for=None,
                alpha=0.05,
                confidence_interval_width=None,
                sample_size=2,
                target_power=None,
                selected_tests=None,
                gaussian_covariate= None,
                scale_factor=None,
                variance_scale_factor=None,
                power_curve=None)
        actual = StudyDesign()
        self.assertEqual(vars(expected), vars(actual))

    def test___str__(self):
        """Should print a statement containing name and values as a list"""
        expected = False
        if self.m.name in str(self.m) and str(self.m.matrix) in str(self.m):
            expected = True
        self.assertTrue(expected)

    def test_load_from_json(self):
        """Should read the study design correctly from the model on model_1.json"""
        outcome_1 = Outcome(name='one')
        outcome_2 = Outcome(name='teo')
        rep_meas_1 = RepeatedMeasure(name='repMeas', values=[0, 1], units='time', type='Numeric', partial_u_matrix=np.matrix([[1],[-1]]), correlation_matrix=np.matrix([[1, 0],[0, 1]]))
        cluster_1 = Cluster(name='clstr', levels=[ClusterLevel(level_name='1'), ClusterLevel(level_name='2', no_elements=2)])
        predictor_1 = Predictor(name='prdctr', values=['grp1', 'grp2'])

        isu_factors = IsuFactors(variables=[outcome_1, outcome_2, rep_meas_1, cluster_1, predictor_1],
                                 smallest_group_size=2,
                                 outcome_correlation_matrix=np.matrix([[1, 0], [0, 1]]),
                                 outcome_repeated_measure_st_devs=[
                                     OutcomeRepeatedMeasureStDev(outcome='one', repeated_measure='repMeas', values=[2, 3]),
                                     OutcomeRepeatedMeasureStDev(outcome='teo', repeated_measure='repMeas', values=[4, 5])])
        
        expected = StudyDesign(isu_factors=isu_factors,
                               target_event=TargetEvent.REJECTION,
                               target_power=0.5,
                               solve_for=SolveFor.POWER,
                               confidence_interval_width=1,
                               sample_size=10,
                               selected_tests=[Tests.HOTELLING_LAWLEY, Tests.PILLAI_BARTLET, Tests.WILKS_LIKLIEHOOD],
                               gaussian_covariate={'variance': 1},
                               scale_factor=1,
                               variance_scale_factor=[3, 4],
                               power_curve=PowerCurve(confidence_interval=ConfidenceInterval(assumptions='Beta Fixed',
                                                                                             beta_sample_size=10),
                                                      x_axis='DesiredPower',
                                                      data_series=[DataSeries(variance_scale_factor=3)]))

        json_data = open("model_2.json")
        data = json_data.read()
        json_data.close()
        actual = StudyDesign().load_from_json(data)
        model = LinearModel()
        model.from_study_design(actual)
        power = unirep._chi_muller_muller_barton_1989(sigma_star=model.sigma_star,
                                                      rank_U=np.linalg.matrix_rank(model.u_matrix),
                                                      total_N=20,
                                                      rank_X=np.linalg.matrix_rank(model.essence_design_matrix))

        self.assertEqual(power, 0.96847792614988382)


if __name__ == '__main__':
    unittest.main()
