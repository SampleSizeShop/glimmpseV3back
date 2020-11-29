from app.calculation_service.model.scenario_inputs import ScenarioInputs
from app.calculation_service.model.study_design import StudyDesign
from app.calculation_service.api import _generate_models, _calculate_power
import pandas as pd
import numpy as np
import xml.etree.ElementTree as et
import timeit

pd.set_option('precision', 7)

def json_power(json_path):
    with open(json_path, 'r') as f:
        data = f.read()
    inputs = ScenarioInputs().load_from_json(data)
    scenario = StudyDesign().load_from_json(data)
    models = _generate_models(scenario, inputs)
    results = []
    for m in models:
        result = _calculate_power(m)
        outdata = {'Power': result['power'],
                   'Test': result['test'],
                   'Sigma Scale': result['model']['variance_scale_factor'],
                   'Beta Scale': result['model']['means_scale_factor'],
                   'Total N': result['model']['total_n'],
                   'Alpha': result['model']['alpha']}
        results.append(outdata)

    return pd.DataFrame(results)


# def parse_XML(xml_file, df_cols):
#     """Parse the input XML file and store the result in a pandas
#     DataFrame with the given columns.
#
#     The first element of df_cols is supposed to be the identifier
#     variable, which is an attribute of each node element in the
#     XML data; other features will be parsed from the text content
#     of each sub-element.
#     """
#
#     xtree = et.parse(xml_file)
#     xroot = xtree.getroot()
#     rows = []
#
#     for node in xroot:
#         res = []
#         for col in df_cols:
#             res.append(node.attrib.get(col))
#         rows.append({df_cols[i]: res[i]
#                      for i, _ in enumerate(df_cols)})
#
#     out_df = pd.DataFrame(rows, columns=df_cols)
#
#     return out_df


df_vtest1 = json_power(r'./testInputs/Test01_V3_ConditionalTwoSampleTTest.json')
df_vtest1.head()
df_vtest1.shape

df_v2results = pd.read_csv(r'./testInputs/Test01_V3_ConditionalTwoSampleTTest_v2results.csv')
df_v2results.head()
df_v2results.shape

df_results1 = pd.merge(df_vtest1, df_v2results, how='outer', left_on=['Sigma Scale', 'Beta Scale', 'Total N', 'Alpha'], right_on=['Sigma Scale', 'Beta Scale', 'Total N', 'Alpha'])
df_results1['deviation_sas_v3'] = abs(df_results1.Power - df_results1.SAS_Power).apply(lambda x: '%.7f' % x)
df_results1['deviation_sim_v3'] = abs(df_results1.Power - df_results1.Sim_Power).apply(lambda x: '%.7f' % x)
df_results1 = df_results1.sort_values(by=['Sigma Scale', 'Beta Scale', 'Total N', 'Alpha'])
df_results1 = df_results1.round({'Power':7, 'SAS_Power':7, 'deviation_sas_v3':7, 'Sim_Power':7, 'deviation_sim_v3':7})
df_results1.Power = df_results1.Power.apply('{:0<9}'.format)
df_results1.SAS_Power = df_results1.SAS_Power.apply('{:0<9}'.format)
df_results1.deviation_sas_v3 = df_results1.deviation_sas_v3.apply('{:0<9}'.format)
df_results1.Sim_Power = df_results1.Sim_Power.apply('{:0<9}'.format)
df_results1.deviation_sim_v3 = df_results1.deviation_sim_v3.apply('{:0<9}'.format)
df_results1["SAS Power (deviation)"] = df_results1["SAS_Power"].astype('str') + " (" + df_results1["deviation_sas_v3"].astype('str') + ")"
df_results1["Sim Power (deviation)"] = df_results1["Sim_Power"].astype('str') + " (" + df_results1["deviation_sim_v3"].astype('str') + ")"
df_results1_out = df_results1[['Power', 'SAS Power (deviation)', 'Sim Power (deviation)', 'Test_x', 'Sigma Scale', 'Beta Scale', 'Total N', 'Alpha']]
# print(df_results1_out)
print(df_results1_out.to_latex(index=False))


# df_sasxmloutput = parse_XML(r'C:\Users\liqian\Dropbox (UFL)\Project_QL\2017_GLIMMPSE_V3\PyGLIMMPSE\TEXT\Test01_V3_ConditionalTwoSampleTTest_v2results.xml', ['test','alpha','nominalPower','actualPower','betaScale','sigmaScale','sampleSize','powerMethod'])
# df_sasxmloutput.alpha = df_sasxmloutput.alpha.astype('float64')
# df_sasxmloutput.nominalPower = df_sasxmloutput.nominalPower.astype('float64')
# df_sasxmloutput.actualPower = df_sasxmloutput.actualPower.astype('float64')
# df_sasxmloutput.betaScale = df_sasxmloutput.betaScale.astype('float64')
# df_sasxmloutput.sigmaScale = df_sasxmloutput.sigmaScale.astype('float64')
# df_sasxmloutput.sampleSize = df_sasxmloutput.sampleSize.astype('int64')
#
# # print(df_vtest1.dtypes)
# # print(df_sasxmloutput.dtypes)
#
# df_results2 = pd.merge(df_vtest1, df_sasxmloutput, how='outer', left_on=['Sigma Scale', 'Beta Scale', 'Total N', 'Alpha'], right_on=['sigmaScale', 'betaScale', 'sampleSize', 'alpha'])
# df_results2['deviation_sas_v3'] = df_results2.Power - df_results2.actualPower