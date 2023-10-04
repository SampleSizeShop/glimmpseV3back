import math
import os
import platform
import subprocess
import json

from time import perf_counter

from app.calculation_service.model.scenario_inputs import ScenarioInputs
from app.calculation_service.model.study_design import StudyDesign
from app.calculation_service.api import _generate_models, _calculate_power
import pandas as pd
import numpy as np
from time import perf_counter
from typing import List
import scipy.stats as stats

def sstar_desc(is_gaussian: bool):
    sstar_desc = '''$\\mathbf{\\Sigma}_{*} = (\\mathbf{U}\'_{o} \\mathbf{\\Sigma}_o \\mathbf{U}_o)
                            \\otimes (\\mathbf{U}_r\' \\mathbf{\\Sigma}_r \\mathbf{U}_r)
                            \\otimes (\\mathbf{U}_c\' \\mathbf{\\Sigma}_c \\mathbf{U}_c)'''

    if is_gaussian:
        sstar_desc = sstar_desc + ' - \\mathbf{U}\'\\mathbf{\\Sigma}_{yg}\\sigma_{g}^{-2}\\mathbf{\\Sigma}_{yg}\'\\mathbf{U}'
    sstar_desc += ' = '
    return sstar_desc

def array_to_matrix(arr: List[List[float]]):
    arr = np.asarray(arr)
    latex = '\\begin{bmatrix} \n'
    for row in arr:
        for i, col in enumerate(row):
            latex = latex + ' ' + str(col) + row_end(i, row)
    latex = latex + '\\end{bmatrix}'
    return latex

def row_end(i, row):
    if i == len(row) - 1:
        return ' \\\\ \n'
    return ' & '

def write_pdf(tex_filename):
    filename, ext = os.path.splitext(tex_filename)
    # the corresponding PDF filename
    pdf_filename = filename + '.pdf'

    # compile TeX file
    subprocess.run(['pdflatex', '-interaction=nonstopmode', tex_filename])

    # check if PDF is successfully generated
    if not os.path.exists(pdf_filename):
        raise RuntimeError('PDF output not found')

    # open PDF with platform-specific command
    if platform.system().lower() == 'darwin':
        subprocess.run(['open', pdf_filename])
    elif platform.system().lower() == 'windows':
        os.startfile(pdf_filename)
    elif platform.system().lower() == 'linux':
        subprocess.run(['xdg-open', pdf_filename])
    else:
        raise RuntimeError('Unknown operating system "{}"'.format(platform.system()))

def write_tex_file(is_gaussian, filename, title, introduction, description, list_inputs, timings_table, deviations_table, results_table):
    f = open(filename + ".tex", "w")
    f.write("\\documentclass{article}\n")
    f.write("\\usepackage{geometry}\n")
    f.write("\\usepackage{amsmath}\n")
    f.write("\\geometry{legalpaper, landscape, margin = 0.25 in }\n")
    f.write("\\usepackage{booktabs}\n")
    f.write("\\usepackage{longtable}\n")
    f.write("\\title{" + title + "}\n")
    f.write("\\author{Alasdair Macleod}\n")
    f.write("\\begin{document}\n")
    f.write("\\maketitle")
    f.write("\n")
    f.write("\\section{Introduction}")
    f.write("\n")
    f.write(introduction)
    f.write("\n")
    f.write("\\section{Study Design}")
    f.write("\n")
    f.write(description)
    f.write("\n")
    f.write("\\subsection{Inputs}")
    f.write("\n")
    f.write("\\subsubsection{Type One Error Rates}")
    f.write("\n")
    f.write(list_inputs["_typeOneErrorRate"])
    f.write("\n")
    f.write("\\subsubsection{Means Scale Factors}")
    f.write("\n")
    f.write(list_inputs["_scaleFactor"])
    f.write("\n")
    f.write("\\subsubsection{Variance Scale Factors}")
    f.write("\n")
    f.write(list_inputs["_varianceScaleFactors"])
    f.write("\n")
    f.write("\\subsubsection{Per Group Sample Size}")
    f.write("\n")
    f.write(list_inputs["smallestGroupSize"])
    f.write("\n")
    f.write("\\subsubsection{Power Method}")
    f.write("\n")
    f.write(list_inputs["_powerMethod"])
    f.write("\n")
    f.write("\\subsubsection{Tests}")
    f.write("\n")
    f.write(list_inputs["_selectedTests"])
    f.write("\n")
    f.write("\\subsubsection{Matrices}")
    f.write("\n")
    f.write(list_inputs["_essenceX"])
    f.write("\n")
    f.write("\n")
    f.write(list_inputs["_C"])
    f.write("\n")
    f.write("\n")
    f.write(list_inputs["_B"])
    f.write("\n")
    f.write("\n")
    f.write(list_inputs["_U"])
    f.write("\n")
    f.write("\n")
    f.write(sstar_desc(is_gaussian))
    f.write("\n")
    f.write("\n")
    f.write(array_to_matrix(list_inputs["_Sigma_Star"]))
    f.write("\n")
    f.write("\n")
    f.write(list_inputs["_ThetaO"])
    f.write("\n")
    f.write("\n")
    f.write(list_inputs["_Theta"])
    f.write("\n")
    f.write("\n")
    f.write(list_inputs["_M"])
    f.write("\n")
    f.write("\n")
    f.write(list_inputs["_nuE"])
    f.write("\n")
    f.write("\n")
    f.write(list_inputs["_repN"])
    f.write("\n")
    f.write("\n")
    f.write(list_inputs["_D"])
    f.write("\n")
    f.write("\\section{Validation Results}")
    f.write("\n")
    f.write("\\subsection{Timings}")
    f.write("\n")
    f.write(timings_table.replace("\\begin{table}", "\\begin{table} \\n \\centering"))
    f.write("\n")
    f.write("\\subsection{Summary Statistics}")
    f.write("\n")
    f.write(deviations_table.replace("\\begin{table}", "\\begin{table} \\n \\centering"))
    f.write("\n")
    f.write("\\subsection{Full Validation Results}")
    f.write("\n")
    f.write(results_table.replace("\\begin{table}", "\\begin{table} \\n \\centering"))
    f.write("\n")
    f.write("\\section{References}")
    f.write("\\end{document}")
    f.close()

def write_tex_results_file(table):
    f = open("V3ResultsTable.tex", "w")
    f.write("\\documentclass{article}\n")
    f.write("\\usepackage{geometry}\n")
    f.write("\\usepackage{amsmath}\n")
    f.write("\\geometry{legalpaper, landscape, margin = 0.25 in }\n")
    f.write("\\usepackage{booktabs}\n")
    f.write("\\usepackage{longtable}\n")
    f.write("\\title{Glimmpse V3 Validation Results}\n")
    f.write("\\author{Alasdair Macleod}\n")
    f.write("\\begin{document}\n")
    f.write("\\maketitle")
    f.write("\n")
    f.write("\\section{Validation Results}")
    f.write("\n")
    f.write("\\subsection{Full Validation Results}")
    f.write("\n")
    f.write(table.to_latex(index=False).replace("\\begin{table}", "\\begin{table} \\n \\centering"))
    f.write("\n")
    f.write("\\end{document}")
    f.close()

pd.set_option('precision', 7)


def json_power(json_path):
    with open(json_path, 'r') as f:
        data = f.read()
    inputs = ScenarioInputs().load_from_json(data)
    scenario = StudyDesign().load_from_json(data)
    models = _generate_models(scenario, inputs)
    results = []
    for m in models:
        t1 = perf_counter()
        result = _calculate_power(m)
        t2 = perf_counter()
        outdata = {'Power': result['power'],
                   'Test': result['test'],
                   'Sigma Scale': result['model']['variance_scale_factor'],
                   'Beta Scale': result['model']['means_scale_factor'],
                   'Total N': result['model']['total_n'],
                   'Alpha': result['model']['alpha'],
                   'Time': t2-t1}
        results.append(outdata)

    return pd.DataFrame(results)

def json_power_with_confidence_intervals(json_path):
    with open(json_path, 'r') as f:
        data = f.read()
    inputs = ScenarioInputs().load_from_json(data)
    scenario = StudyDesign().load_from_json(data)
    models = _generate_models(scenario, inputs)
    results = []
    for m in models:
        t1 = perf_counter()
        result = _calculate_power(m)
        t2 = perf_counter()
        outdata = {'Power': result['power'],
                   'Lower bound v3': result['lower_bound'],
                   'Upper bound v3': result['upper_bound'],
                   'Test': result['test'],
                   'Sigma Scale': result['model']['variance_scale_factor'],
                   'Beta Scale': result['model']['means_scale_factor'],
                   'Total N': result['model']['total_n'],
                   'Alpha': result['model']['alpha'],
                   'Time': t2-t1}
        results.append(outdata)

    return pd.DataFrame(results)

def json_power_by_delta(json_path):
    deltas = [x * 0.0008 for x in range(1, 251, 1)]
    top = [x for x in range(0, 5, 1)]
    bottom = [x for x in range(5, 10, 1)]

    with open(json_path, 'r') as f:
        data = f.read()
    results = []

    for d in deltas:

        inputs = ScenarioInputs().load_from_json(data)
        scenario = StudyDesign().load_from_json(data)


        for row in top:
            cell = scenario.isu_factors.marginal_means[0].get('_table')[row][2]
            cell['value'] = cell['value'] + d
        for row in bottom:
            cell = scenario.isu_factors.marginal_means[0].get('_table')[row][2]
            cell['value'] = cell['value'] - d
        models = _generate_models(scenario, inputs)
        for m in models:
            m.errors = []
            t1 = perf_counter()
            result = _calculate_power(m)
            t2 = perf_counter()
            outdata = {'Power': result['power'],
                       'Lower bound v3': result['lower_bound'],
                       'Upper bound v3': result['upper_bound'],
                       'Test': result['test'],
                       'Sigma Scale': result['model']['variance_scale_factor'],
                       'Beta Scale': result['model']['means_scale_factor'],
                       'Total N': result['model']['total_n'],
                       'Alpha': result['model']['alpha'],
                        'Time': t2-t1}
            results.append(outdata)

    return pd.DataFrame(results)

def get_summary_results(V2_results, _df_v2results, file_path):
    length = _df_v2results.shape[0] + 3
    _df_v2summary = pd.read_csv(file_path + V2_results, skiprows=length, engine='python', sep=':')
    _df_v2summary = _df_v2summary.rename(columns={_df_v2summary.columns[0]: 'Name'})
    return _df_v2summary


def get_print_output(_df_v2results, _df_vtest, output_name):
    _df_output = pd.merge(_df_vtest, _df_v2results, how='outer',
                          left_on=['Sigma Scale', 'Beta Scale', 'Total N', 'Alpha'],
                          right_on=['Sigma Scale', 'Beta Scale', 'Total N', 'Alpha'])
    _df_output = _df_output[_df_output['SAS_Power'].notna()]
    _df_output['deviation_sas_v3'] = abs(_df_output.Power - _df_output.SAS_Power).apply(lambda x: '%.7f' % x)
    _df_output['deviation_sim_v3'] = abs(_df_output.Power - _df_output.Sim_Power).apply(lambda x: '%.7f' % x)
    _df_output['deviation_v2_v3'] = abs(_df_output.Power - _df_output.Power_v2).apply(lambda x: '%.7f' % x)
    _df_output['lower_bound'] = calc_lower_bound(_df_output.Power).apply(lambda x: '%.7f' % x)
    _df_output['upper_bound'] = calc_upper_bound(_df_output.Power).apply(lambda x: '%.7f' % x)
    _df_output = _df_output.sort_values(by=['Sigma Scale', 'Beta Scale', 'Total N', 'Alpha'], ignore_index=True)
    _df_output = _df_output.round(
        {'Power': 7, 'SAS_Power': 7, 'deviation_sas_v3': 7, 'Sim_Power': 7, 'Power_v2': 7, 'deviation_sas_v3': 7,
         'deviation_sim_v3': 7, 'deviation_v2_v3': 7})
    _df_output.Power = _df_output.Power.apply('{:0<9}'.format)
    _df_output.SAS_Power = _df_output.SAS_Power.apply('{:0<9}'.format)
    _df_output.deviation_sas_v3 = _df_output.deviation_sas_v3.apply('{:0<9}'.format)
    _df_output.Sim_Power = _df_output.Sim_Power.apply('{:0<9}'.format)
    _df_output.deviation_sim_v3 = _df_output.deviation_sim_v3.apply('{:0<9}'.format)
    _df_output.Power_v2 = _df_output.Power_v2.apply('{:0<9}'.format)
    _df_output.deviation_v2_v3 = _df_output.deviation_v2_v3.apply('{:0<9}'.format)
    _df_output.to_excel(output_name + '.xlsx')
    _df_output["Test"] = _df_output["Test_x"]
    _df_output["GLIMMPSE V3 Power"] = _df_output["Power"]
    _df_output["SAS Power (deviation)"] = _df_output["SAS_Power"].astype('str') + " (" + _df_output[
        "deviation_sas_v3"].astype('str') + ")"
    _df_output["Sim Power (deviation)"] = _df_output["Sim_Power"].astype('str') + " (" + _df_output[
        "deviation_sim_v3"].astype('str') + ")"
    _df_output["GLIMMPSE V2 Power (deviation)"] = _df_output["Power_v2"].astype('str') + " (" + _df_output[
        "deviation_sim_v3"].astype('str') + ")"
    _df_print = _df_output[
        ['GLIMMPSE V3 Power', 'lower_bound', 'upper_bound','Sim Power (deviation)', 'GLIMMPSE V2 Power (deviation)', 'Test',
         'Sigma Scale', 'Beta Scale', 'Total N',
         'Alpha', 'Time']]
    _df_print = _df_print[_df_print['Test'].notna()]
    _df_output = _df_output[_df_output['Test_x'].notna()]
    return _df_output, _df_print

def get_print_output_additive(_df_v2results, _df_vtest, output_name):
    _df_output = pd.merge(_df_vtest, _df_v2results, how='outer',
                          left_on=['Beta Scale', 'Alpha'],
                          right_on=['Beta Scale', 'Alpha'])
    _df_output = _df_output[_df_output['SAS_Power'].notna()]
    _df_output['deviation_sas_v3'] = abs(_df_output.Power - _df_output.SAS_Power).apply(lambda x: '%.7f' % x)
    _df_output['deviation_sim_v3'] = abs(_df_output.Power - _df_output.Sim_Power).apply(lambda x: '%.7f' % x)
    _df_output['deviation_v2_v3'] = abs(_df_output.Power - _df_output.Power_v2).apply(lambda x: '%.7f' % x)
    _df_output['lower_bound'] = calc_lower_bound(_df_output.Power).apply(lambda x: '%.7f' % x)
    _df_output['upper_bound'] = calc_upper_bound(_df_output.Power).apply(lambda x: '%.7f' % x)
    _df_output = _df_output.sort_values(by=['Beta Scale', 'Alpha'], ignore_index=True)
    _df_output = _df_output.round(
        {'Power': 7, 'SAS_Power': 7, 'deviation_sas_v3': 7, 'Sim_Power': 7, 'Power_v2': 7, 'deviation_sas_v3': 7,
         'deviation_sim_v3': 7, 'deviation_v2_v3': 7})
    _df_output.Power = _df_output.Power.apply('{:0<9}'.format)
    _df_output.SAS_Power = _df_output.SAS_Power.apply('{:0<9}'.format)
    _df_output.deviation_sas_v3 = _df_output.deviation_sas_v3.apply('{:0<9}'.format)
    _df_output.Sim_Power = _df_output.Sim_Power.apply('{:0<9}'.format)
    _df_output.deviation_sim_v3 = _df_output.deviation_sim_v3.apply('{:0<9}'.format)
    _df_output.Power_v2 = _df_output.Power_v2.apply('{:0<9}'.format)
    _df_output.deviation_v2_v3 = _df_output.deviation_v2_v3.apply('{:0<9}'.format)
    _df_output.to_excel(output_name + '.xlsx')
    _df_output["Test"] = _df_output["Test_x"]
    _df_output["GLIMMPSE V3 Power"] = _df_output["Power"]
    _df_output["SAS Power (deviation)"] = _df_output["SAS_Power"].astype('str') + " (" + _df_output[
        "deviation_sas_v3"].astype('str') + ")"
    _df_output["Sim Power (deviation)"] = _df_output["Sim_Power"].astype('str') + " (" + _df_output[
        "deviation_sim_v3"].astype('str') + ")"
    _df_output["GLIMMPSE V2 Power (deviation)"] = _df_output["Power_v2"].astype('str') + " (" + _df_output[
        "deviation_sim_v3"].astype('str') + ")"
    _df_print = _df_output[
        ['GLIMMPSE V3 Power', 'lower_bound', 'upper_bound','Sim Power (deviation)', 'GLIMMPSE V2 Power (deviation)', 'Test', 'Beta Scale',
         'Alpha', 'Time']]
    _df_print = _df_print[_df_print['Test'].notna()]
    _df_output = _df_output[_df_output['Test_x'].notna()]
    return _df_output, _df_print


def get_inputs(V3_JSON, file_path):
    p = [file_path + n for n in V3_JSON]
    with open(p[0], 'r') as f:
        data = f.read()
    inputs = json.loads(data)
    list_inputs = {}
    list_inputs["_typeOneErrorRate"] = str([i for i in inputs["_typeOneErrorRate"]])
    list_inputs["_selectedTests"] = str([i for i in inputs["_selectedTests"]])
    list_inputs["_scaleFactor"] = str([i for i in inputs["_scaleFactor"]])
    list_inputs["_varianceScaleFactors"] = str([i for i in inputs["_varianceScaleFactors"]])
    list_inputs["smallestGroupSize"] = str([i for i in inputs["_isuFactors"]["smallestGroupSize"]])
    if len(inputs["_quantiles"]) > 0:
        list_inputs["_powerMethod"] = "Conditional: Quantiles " + str([i for i in inputs["_quantiles"]])
    else:
        list_inputs["_powerMethod"] = "Unconditional"
    i = ScenarioInputs().load_from_json(data)
    scenario = StudyDesign().load_from_json(data)
    models = _generate_models(scenario, i)
    model = models[0]
    list_inputs["_essenceX"] = "Es(\\mathbf{X}) = " + array_to_matrix(model.essence_design_matrix)
    list_inputs["_B"] = "\\mathbf{B} = " + array_to_matrix(model.hypothesis_beta)
    list_inputs["_C"] = "\\mathbf{C} = " + array_to_matrix(model.c_matrix)
    list_inputs["_U"] = "\\mathbf{U} = " + array_to_matrix(model.u_matrix)
    list_inputs["_M"] = "\\mathbf{M} = " + array_to_matrix(model.m)
    list_inputs["_Sigma_Star"] = model.sigma_star
    list_inputs["_ThetaO"] = "\\mathbf{\\Theta}_{0} = " + array_to_matrix(model.theta_zero)
    list_inputs["_Theta"] = "\\mathbf{\\Theta} = " + array_to_matrix(model.theta)
    list_inputs["_nuE"] = "$\\nu_e = " + str(model.nu_e)
    list_inputs["_repN"] = "Number of repeated rows in design matrix = " + str(model.repeated_rows_in_design_matrix)
    list_inputs["_D"] = "Es(\mathbf{\Delta}) = " + array_to_matrix(model.delta)

    return list_inputs


def tex_table(file_path, output_name, V3_JSON: [], V2_results, v3_table_data:dict=None, v3_example_name:str=None, v3_table=False):
    list_inputs = get_inputs(V3_JSON, file_path)

    _df_vtest = pd.concat([json_power(file_path + model) for model in V3_JSON], ignore_index=True)
    # lower bound upper bound calculation
    _df_v2results = pd.read_csv(file_path + V2_results, skipfooter=9, engine='python',
                                na_values=('NaN', 'n/a', ' n/a', 'nan', 'nan000000'))
    _df_v2summary = get_summary_results(V2_results, _df_v2results, file_path)

    _df_output, _df_print = get_print_output(_df_v2results, _df_vtest, output_name)
    if v3_table:
        v3_table_data.update({v3_example_name: [v3_example_name, _df_output["Power"].get(0), _df_output["Sim_Power"].get(0), _df_output["deviation_sim_v3"].get(0), _df_output["lower_bound"].get(0), _df_output["upper_bound"].get(0)]})
    timings = pd.DataFrame({'Timing for': ['GLIMMPSE V3', 'GLIMMPSE V2', 'Simulation'],
                            'Total': [_df_output["Time"].sum(), _df_v2summary['Name']['Total Calculation Time'],
                                      _df_v2summary['Name']['Total Simulation Time']],
                            'Mean': [_df_output["Time"].mean(), _df_v2summary['Name']['Mean Calculation Time'],
                                     _df_v2summary['Name']['Mean Simulation Time']]})
    deviations = pd.DataFrame({'GLIMMPSE V3 deviation from': ['Simulation', 'SAS', 'GLIMMPSE V2'],
                               'Max. Deviation': [_df_output["deviation_sas_v3"].max(),
                                                  _df_output["deviation_sim_v3"].max(),
                                                  _df_output["deviation_v2_v3"].max()]})

    return timings.to_latex(index=False), deviations.to_latex(index=False), _df_print.to_latex(index=False,
                                                                                               longtable=True), list_inputs, v3_table_data
def tex_table_additive(file_path, output_name, V3_JSON: [], V2_results, v3_table_data:dict=None, v3_example_name:str=None, v3_table=False):
    list_inputs = get_inputs(V3_JSON, file_path)

    _df_vtest = pd.concat([json_power(file_path + model) for model in V3_JSON], ignore_index=True)
    # lower bound upper bound calculation
    _df_v2results = pd.read_csv(file_path + V2_results, skipfooter=9, engine='python',
                                na_values=('NaN', 'n/a', ' n/a', 'nan', 'nan000000'))
    _df_v2summary = get_summary_results(V2_results, _df_v2results, file_path)

    _df_output, _df_print = get_print_output_additive(_df_v2results, _df_vtest, output_name)
    if v3_table:
        v3_table_data.update({v3_example_name: [v3_example_name, _df_output["Power"].get(0),
                                                _df_output["Sim_Power"].get(0), _df_output["deviation_sim_v3"].get(0),
                                                _df_output["lower_bound"].get(0), _df_output["upper_bound"].get(0)]})
    timings = pd.DataFrame({'Timing for': ['GLIMMPSE V3', 'GLIMMPSE V2', 'Simulation'],
                            'Total': [_df_output["Time"].sum(), _df_v2summary['Name']['Total Calculation Time'],
                                      _df_v2summary['Name']['Total Simulation Time']],
                            'Mean': [_df_output["Time"].mean(), _df_v2summary['Name']['Mean Calculation Time'],
                                     _df_v2summary['Name']['Mean Simulation Time']]})
    deviations = pd.DataFrame({'GLIMMPSE V3 deviation from': ['Simulation', 'SAS', 'GLIMMPSE V2'],
                               'Max. Deviation': [_df_output["deviation_sas_v3"].max(),
                                                  _df_output["deviation_sim_v3"].max(),
                                                  _df_output["deviation_v2_v3"].max()]})

    return timings.to_latex(index=False), deviations.to_latex(index=False), _df_print.to_latex(index=False,
                                                                                               longtable=True), list_inputs, v3_table_data


def calc_lower_bound(pow):
    z = stats.norm.ppf(0.975)
    r = 10000
    # s = [math.sqrt(p * (1 - p) / r) for p in pow]
    return pd.Series([p - (z * (math.sqrt(p * (1 - p) / r))) for p in pow])

def calc_upper_bound(pow):
    z = stats.norm.ppf(0.975)
    r = 10000
    # s = [math.sqrt(p * (1 - p) / r) for p in pow]
    return pd.Series([p + (z * (math.sqrt(p * (1 - p) / r))) for p in pow])


def tex_table_test_4(file_path, output_name, V3_JSON: [], V2_results):
    list_inputs = get_inputs(V3_JSON, file_path)
    _df_vtest = pd.concat([json_power_with_confidence_intervals(file_path + model) for model in V3_JSON], ignore_index=True).sort_values(['Beta Scale', 'Total N'], ignore_index=True).add_suffix("_v3")
    _df_v2results = pd.read_csv(file_path + V2_results,skipfooter=9, engine='python', na_values=('NaN', 'n/a', ' n/a', 'nan', 'nan000000'))
    _df_v2results = _df_v2results[_df_v2results['Beta Scale'] != 0]
    _df_v2results = _df_v2results[_df_v2results['Beta Scale'].notna()]
    _df_v2results = _df_v2results.sort_values(['Beta Scale', 'Total N'], ignore_index=True)

    _df_v2summary = get_summary_results(V2_results, _df_v2results, file_path)

    _df_output, _df_print = get_print_output_with_concat(_df_v2results, _df_vtest, output_name, True)

    timings = pd.DataFrame({'Timing for': ['GLIMMPSE V3', 'GLIMMPSE V2', 'Simulation'],
                            'Total': [_df_output["Time"].sum(), _df_v2summary['Name']['Total Calculation Time'],
                                      _df_v2summary['Name']['Total Simulation Time']],
                            'Mean': [_df_output["Time"].mean(), _df_v2summary['Name']['Mean Calculation Time'],
                                     _df_v2summary['Name']['Mean Simulation Time']]})
    deviations = pd.DataFrame({'GLIMMPSE V3 deviation from': ['lower confidence interval v2', 'upper confidence interval v2', 'Simulation', 'SAS', 'GLIMMPSE V2'],
                               'Max. Deviation': [_df_output["deviation_lower_v2_v3"].max(),
                                                  _df_output["deviation_upper_v2_v3"].max(),
                                                  _df_output["deviation_sas_v3"].max(),
                                                  _df_output["deviation_sim_v3"].max(),
                                                  _df_output["deviation_v2_v3"].max()]})

    return timings.to_latex(index=False), deviations.to_latex(index=False), _df_print.to_latex(index=False,
                                                                                               longtable=True), list_inputs


def get_print_output_with_concat(_df_v2results, _df_vtest, output_name, confidence_limits=False):
    _df_output = pd.concat([_df_vtest, _df_v2results], axis=1)
    _df_output['deviation_sas_v3'] = abs(_df_output.Power_v3 - _df_output.SAS_Power).apply(lambda x: '%.7f' % x)
    _df_output['deviation_sim_v3'] = abs(_df_output.Power_v3 - _df_output.Sim_Power).apply(lambda x: '%.7f' % x)
    _df_output["Power_v2"] = pd.to_numeric(_df_output["Power_v2"], downcast="float")
    _df_output['deviation_v2_v3'] = abs(_df_output.Power_v3 - _df_output.Power_v2).apply(lambda x: '%.7f' % x)
    _df_output = _df_output.round(
        {'Power_v3': 7, 'SAS_Power': 7, 'deviation_sas_v3': 7, 'Sim_Power': 7, 'Power_v2': 7, 'deviation_sas_v3': 7,
         'deviation_sim_v3': 7, 'deviation_v2_v3': 7})
    _df_output.Power_v3 = _df_output.Power_v3.apply('{:0<9}'.format)
    _df_output.SAS_Power = _df_output.SAS_Power.apply('{:0<9}'.format)
    _df_output.deviation_sas_v3 = _df_output.deviation_sas_v3.apply('{:0<9}'.format)
    _df_output.Sim_Power = _df_output.Sim_Power.apply('{:0<9}'.format)
    _df_output.deviation_sim_v3 = _df_output.deviation_sim_v3.apply('{:0<9}'.format)
    _df_output["Power_v2"] = _df_output["Power_v2"].apply(lambda x: '%.7f' % x)
    _df_output.Power_v2 = _df_output.Power_v2.apply('{:0<9}'.format)
    _df_output.deviation_v2_v3 = _df_output.deviation_v2_v3.apply('{:0<9}'.format)
    _df_output.to_excel(output_name + '.xlsx')
    _df_output["Test"] = _df_output["Test_v3"]
    _df_output["GLIMMPSE V3 Power"] = _df_output["Power_v3"]
    _df_output["SAS Power (deviation)"] = _df_output["SAS_Power"].astype('str') + " (" + _df_output[
        "deviation_sas_v3"].astype('str') + ")"
    _df_output["Sim Power (deviation)"] = _df_output["Sim_Power"].astype('str') + " (" + _df_output[
        "deviation_sim_v3"].astype('str') + ")"
    _df_output["GLIMMPSE V2 Power (deviation)"] = _df_output["Power_v2"].astype('str') + " (" + _df_output[
        "deviation_sim_v3"].astype('str') + ")"
    if confidence_limits:
        _df_output['deviation_lower_v2_v3'] = abs(_df_output[_df_output.columns[1]] - _df_output.lower_v2).apply(lambda x: '%.7f' % x)
        _df_output['deviation_upper_v2_v3'] = abs(_df_output[_df_output.columns[2]] - _df_output.upper_v2).apply(lambda x: '%.7f' % x)
        _df_output[_df_output.columns[1]] = _df_output[_df_output.columns[1]].apply(lambda x: '%.7f' % x)
        _df_output[_df_output.columns[2]] = _df_output[_df_output.columns[2]].apply(lambda x: '%.7f' % x)

        _df_output["GLIMMPSE V3 lower (deviation)"] = _df_output[_df_output.columns[1]].astype('str') + \
                                                      " (" + _df_output["deviation_lower_v2_v3"].astype('str') + ")"
        _df_output["GLIMMPSE V3 upper (deviation)"] = _df_output[_df_output.columns[2]].astype('str') + \
                                                      " (" + _df_output["deviation_upper_v2_v3"].astype('str') + ")"

    _df_print = _df_output[
        ['GLIMMPSE V3 Power', 'Sim Power (deviation)', 'GLIMMPSE V2 Power (deviation)', 'Test',
         'Sigma Scale', 'Beta Scale', 'Total N',
         'Alpha', 'Time_v3']]
    if confidence_limits:
        _df_print = _df_output[
        ['GLIMMPSE V3 Power', 'GLIMMPSE V3 lower (deviation)', 'GLIMMPSE V3 upper (deviation)', 'Sim Power (deviation)', 'GLIMMPSE V2 Power (deviation)', 'lower_v2', 'upper_v2', 'Test',
         'Sigma Scale', 'Beta Scale', 'Total N',
         'Alpha', 'Time_v3']]
    _df_print = _df_print[_df_print['Test'].notna()]
    _df_output = _df_output[_df_output['Test_v3'].notna()]
    _df_output["Time"] = _df_output["Time_v3"]
    return _df_output, _df_print


def tex_table_test_5(file_path, output_name, V3_JSON: [], V2_results):
    list_inputs = get_inputs(V3_JSON, file_path)
    _df_vtest = pd.concat([json_power(file_path + model) for model in V3_JSON], ignore_index=True).sort_values(['Test','Sigma Scale','Beta Scale', 'Total N'], ignore_index=True).add_suffix("_v3")
    _df_v2results = pd.read_csv(file_path + V2_results,skipfooter=9, engine='python', na_values=('NaN', 'n/a', ' n/a'))
    _df_v2results = _df_v2results[_df_v2results['Beta Scale'] != 0]
    _df_v2results = _df_v2results[_df_v2results['Beta Scale'].notna()]
    _df_v2results = _df_v2results.sort_values(['Test','Sigma Scale','Beta Scale', 'Total N'], ignore_index=True)
    _df_output = pd.concat([_df_vtest, _df_v2results], axis=1)

    length = 143
    _df_v2summary = pd.read_csv(file_path + V2_results, skiprows=length, engine='python', sep=':')
    _df_v2summary = _df_v2summary.rename(columns={_df_v2summary.columns[0]: 'Name'})
    _df_output, _df_print = get_print_output_with_concat(_df_v2results, _df_vtest, output_name)

    timings = pd.DataFrame({'Timing for': ['GLIMMPSE V3', 'GLIMMPSE V2', 'Simulation'],
                            'Total': [_df_output["Time"].sum(), _df_v2summary['Name']['Total Calculation Time'],
                                      _df_v2summary['Name']['Total Simulation Time']],
                            'Mean': [_df_output["Time"].mean(), _df_v2summary['Name']['Mean Calculation Time'],
                                     _df_v2summary['Name']['Mean Simulation Time']]})
    deviations = pd.DataFrame({'GLIMMPSE V3 deviation from': ['Simulation', 'SAS', 'GLIMMPSE V2'],
                               'Max. Deviation': [_df_output["deviation_sas_v3"].max(),
                                                  _df_output["deviation_sim_v3"].max(),
                                                  _df_output["deviation_v2_v3"].max()]})

    return timings.to_latex(index=False), deviations.to_latex(index=False), _df_print.to_latex(index=False,
                                                                                               longtable=True), list_inputs


def tex_table_test7(file_path, output_name, V3_JSON: [], V2_results):
    list_inputs = get_inputs(V3_JSON, file_path)
    _df_vtest = pd.concat([json_power(file_path + model) for model in V3_JSON], ignore_index=True).add_suffix("_v3")
    _df_vtest = _df_vtest.sort_values(['Test_v3', 'Total N_v3'], ignore_index=True)
    _df_v2results = pd.read_csv(file_path + V2_results,skipfooter=9, engine='python', na_values=('NaN', 'n/a', ' n/a'))
    _df_v2results = _df_v2results.sort_values(['Test', 'Total N'], ignore_index=True)
    _df_output = pd.concat([_df_vtest, _df_v2results], axis=1)

    _df_output['deviation_sas_v3'] = abs(_df_output.Power_v3 - _df_output.SAS_Power)
    _df_output['deviation_sas_v2_calc'] = abs(_df_output.Power_v2 - _df_output.SAS_Power)
    _df_output['deviation_sim_v3'] = abs(_df_output.Power_v3 - _df_output.Sim_Power)
    _df_output['deviation_sim_v2_calc'] = abs(_df_output.Power_v2 - _df_output.Sim_Power)
    _df_output['deviation_v2_v3'] = abs(_df_output.Power_v3 - _df_output.Power_v2)

    _df_output.to_excel(output_name + '.xlsx')


    length = 24
    _df_v2summary = pd.read_csv(file_path + V2_results, skiprows=length, engine='python', sep=':')
    _df_v2summary = _df_v2summary.rename(columns={_df_v2summary.columns[0]: 'Name'})
    _df_output, _df_print = get_print_output_with_concat(_df_v2results, _df_vtest, output_name)

    timings = pd.DataFrame({'Timing for': ['GLIMMPSE V3', 'GLIMMPSE V2', 'Simulation'],
                            'Total': [_df_output["Time"].sum(), _df_v2summary['Name']['Total Calculation Time'],
                                      _df_v2summary['Name']['Total Simulation Time']],
                            'Mean': [_df_output["Time"].mean(), _df_v2summary['Name']['Mean Calculation Time'],
                                     _df_v2summary['Name']['Mean Simulation Time']]})
    deviations = pd.DataFrame({'GLIMMPSE V3 deviation from': ['Simulation', 'SAS', 'GLIMMPSE V2'],
                               'Max. Deviation': [_df_output["deviation_sim_v3"].max(),
                                                  _df_output["deviation_sas_v3"].max(),
                                                  _df_output["deviation_v2_v3"].max()]})

    return timings.to_latex(index=False), deviations.to_latex(index=False), _df_print.to_latex(index=False,
                                                                                               longtable=True), list_inputs


def tex_table_test_9(file_path, output_name, V3_JSON: [], V2_results):
    _df_vtest = pd.concat([json_power(file_path + model) for model in V3_JSON], ignore_index=True)
    _df_vtest = _df_vtest.sort_values(['Test', 'Total N'], ignore_index=True)
    _df_v2results = pd.read_csv(file_path + V2_results,skipfooter=9, engine='python', na_values=('NaN', 'n/a', ' n/a'))
    _df_v2results = _df_v2results.sort_values(['Test', 'Total N'], ignore_index=True)
    _df_output = pd.concat([_df_vtest, _df_v2results], axis=1)

    _df_output['deviation_sas_v3'] = abs(_df_output.Power - _df_output.SAS_Power)
    _df_output['deviation_sas_v2_calc'] = abs(_df_output.Power_v2 - _df_output.SAS_Power)
    _df_output['deviation_sim_v3'] = abs(_df_output.Power - _df_output.Sim_Power)
    _df_output['deviation_sim_v2_calc'] = abs(_df_output.Power_v2 - _df_output.Sim_Power)
    _df_output['deviation_v2_v3'] = abs(_df_output.Power - _df_output.Power_v2)

    _df_output.to_excel(output_name + '.xlsx')
    _df_output["SAS Power (deviation)"] = _df_output["SAS_Power"].astype('str') + " (" + _df_output[
        "deviation_sas_v3"].astype('str') + ")"
    _df_output["Sim Power (deviation)"] = _df_output["Sim_Power"].astype('str') + " (" + _df_output[
        "deviation_sim_v3"].astype('str') + ")"
    _df_print = _df_output[
        ['Power', 'Sim Power (deviation)', 'Test_x', 'Sigma Scale', 'Beta Scale', 'Total N',
         'Alpha', 'Time']]

    _df_print = _df_print[_df_print['Test_x'].notna()]
    return _df_print.to_latex(index=False)

def tex_table_gaussian(file_path, output_name, V3_JSON: [], V2_results):
    list_inputs = get_inputs(V3_JSON, file_path)
    _df_vtest = pd.concat([json_power_with_confidence_intervals(file_path + model) for model in V3_JSON],
                          ignore_index=True).sort_values(['Test', 'Total N', 'Power'], ignore_index=True).add_suffix("_v3")
    _df_v2results = pd.read_csv(file_path + V2_results, skipfooter=9, engine='python',
                                na_values=('NaN', 'n/a', ' n/a'))
    _df_v2results = _df_v2results[_df_v2results['Test'] != 0]
    _df_v2results = _df_v2results[_df_v2results['Total N'].notna()]
    _df_v2results = _df_v2results.sort_values(['Test', 'Total N', 'Power_v2'], ignore_index=True)
    _df_output = pd.concat([_df_vtest, _df_v2results], axis=1)

    _df_output['deviation_v2_v3'] = abs(_df_output.Power_v3 - _df_output.Power_v2).apply(lambda x: '%.7f' % x)
    _df_output['deviation_sas_v3'] = abs(_df_output.Power_v3 - _df_output.SAS_Power).apply(lambda x: '%.7f' % x)
    _df_output['deviation_sim_v3'] = abs(_df_output.Power_v3 - _df_output.Sim_Power).apply(lambda x: '%.7f' % x)


    _df_output.to_excel(output_name + '.xlsx')

    _df_v2summary = get_summary_results(V2_results, _df_v2results, file_path)
    _df_output, _df_print = get_print_output_with_concat(_df_v2results, _df_vtest, output_name)

    timings = pd.DataFrame({'Timing for': ['GLIMMPSE V3', 'GLIMMPSE V2', 'Simulation'],
                            'Total': [_df_output["Time"].sum(), _df_v2summary['Name']['Total Calculation Time'],
                                      _df_v2summary['Name']['Total Simulation Time']],
                            'Mean': [_df_output["Time"].mean(), _df_v2summary['Name']['Mean Calculation Time'],
                                     _df_v2summary['Name']['Mean Simulation Time']]})
    deviations = pd.DataFrame({'GLIMMPSE V3 deviation from': ['Simulation', 'SAS', 'GLIMMPSE V2'],
                               'Max. Deviation': [_df_output["deviation_sas_v3"].max(),
                                                  _df_output["deviation_sim_v3"].max(),
                                                  _df_output["deviation_v2_v3"].max()]})

    return timings.to_latex(index=False), deviations.to_latex(index=False), _df_print.to_latex(index=False,
                                                                                               longtable=True), list_inputs

def tex_table_by_delta(file_path, output_name, V3_JSON: [], V2_results):
    list_inputs = get_inputs(V3_JSON, file_path)
    _df_vtest = pd.concat([json_power_by_delta(file_path + model) for model in V3_JSON], ignore_index=True).add_suffix("_v3")
    _df_v2results = pd.read_csv(file_path + V2_results,skipfooter=11, engine='python', na_values=('nan', 'NaN', 'n/a', ' n/a'))
    _df_output=pd.concat([_df_vtest, _df_v2results], axis=1)
    _df_output = _df_output[_df_output['SAS_Power'].notna()]
    _df_output.to_excel(output_name + '.xlsx')
    _df_v2summary = get_summary_results(V2_results, _df_v2results, file_path)
    _df_output, _df_print = get_print_output_with_concat(_df_v2results, _df_vtest, output_name, True)

    timings = pd.DataFrame({'Timing for': ['GLIMMPSE V3', 'GLIMMPSE V2', 'Simulation'],
                            'Total': [_df_output["Time"].sum(), _df_v2summary['Name']['Total Calculation Time'],
                                      _df_v2summary['Name']['Total Simulation Time']],
                            'Mean': [_df_output["Time"].mean(), _df_v2summary['Name']['Mean Calculation Time'],
                                     _df_v2summary['Name']['Mean Simulation Time']]})
    deviations = pd.DataFrame({'GLIMMPSE V3 deviation from': ['lower confidence interval v2', 'upper confidence interval v2', 'Simulation', 'SAS', 'GLIMMPSE V2'],
                               'Max. Deviation': [_df_output["deviation_lower_v2_v3"].max(),
                                                  _df_output["deviation_upper_v2_v3"].max(),
                                                  _df_output["deviation_sas_v3"].max(),
                                                  _df_output["deviation_sim_v3"].max(),
                                                  _df_output["deviation_v2_v3"].max()]})

    return timings.to_latex(index=False), deviations.to_latex(index=False), _df_print.to_latex(index=False,
                                                                                               longtable=True), list_inputs


file_path = r'v2TestResults/'

HOMEWORK_1_FILENAME = "Homework1"
HOMEWORK_2_FILENAME = "Homework2"
HOMEWORK_3_FILENAME = "Homework3"
HOMEWORK_4_FILENAME = "Homework4"
HOMEWORK_5_FILENAME = "Homework5"

TEST_1_FILENAME = "Example_1_Power_for_a_two_sample_ttest_for_several_error_variance_values_and_mean_differences"
TEST_2_FILENAME = "Example_2_Power_results_for_a_Paired_Ttest"
TEST_3_FILENAME = "Example_3_Power_for_a_two_sample_ttest_for_various_sample_sizes_and_mean_differences"
TEST_4_FILENAME = "Example_4_Power_and_confidence_limits_for_a_univariate_model"
TEST_5_FILENAME = "Example_5_Power_for_a_test_of_interaction_in_a_multivariate_model"
TEST_6_FILENAME = "Example_6_Power_and_confidence_limits_for_the_univariate_approach_to_repeated_measures_in_a_multivariate_model"
TEST_7_FILENAME = "Example_7_Power_for_a_time_by_treatment_interaction_using_orthogonal_polynomial_contrast_for_time"

GAUSSIAN_TEST_1_FILENAME = "Gaussian_Example_1_Median_power_for_the_HotellingLawley_Trace_using_the_Satterthwaite_approximation"
GAUSSIAN_TEST_4_FILENAME = "Gaussian_Example_4_Unconditional_power_for_the_HotellingLawley_Trace_using_the_Davies"
GAUSSIAN_TEST_5_FILENAME = "Gaussian_Example_5_Median_power_for_the_HotellingLawley_Trace_using_the_Satterthwaite_approximation"
GAUSSIAN_TEST_8_FILENAME = "Gaussian_Example_8 Unconditional_power_for_the_Univariate tests using Davies algorithm"


v3_table_data = dict()
homework1_timings, homework1_deviations, homework1_results, homework1_list_inputs, v3_table_data = tex_table(file_path, TEST_1_FILENAME, ['Homework1.json'], 'Homework1.csv', v3_table_data, '1', True)
homework2_timings, homework2_deviations, homework2_results, homework2_list_inputs, v3_table_data = tex_table(file_path, TEST_2_FILENAME, ['Homework2.json'], 'Homework2.csv', v3_table_data, '2', True)
homework3_timings, homework3_deviations, homework3_results, homework3_list_inputs, v3_table_data = tex_table_additive(file_path, TEST_3_FILENAME, ['Homework3.json'], 'Homework3.csv', v3_table_data, '3', True)
homework4_timings, homework4_deviations, homework4_results, homework4_list_inputs, v3_table_data = tex_table_additive(file_path, TEST_4_FILENAME, ['Homework4.json'], 'Homework4.csv', v3_table_data, '4', True)
homework5_timings, homework5_deviations, homework5_results, homework5_list_inputs, v3_table_data= tex_table(file_path, TEST_5_FILENAME, ['Homework5.json'], 'Homework5.csv', v3_table_data, '5', True)
V3_RESULTS_TABLE = pd.DataFrame.from_dict(v3_table_data, orient='index', columns=['Example #', 'GLIMMPSE V3', 'Simulated Power', 'Deviation', 'Lower Bound', 'Upper Bound'])

OhImAHugeHack = None
test1_timings, test1_deviations, test1_results, test1_list_inputs, OhImAHugeHack = tex_table(file_path, TEST_1_FILENAME, ['Test01_V3_ConditionalTwoSampleTTest.json'], 'Example_1_Power_for_a_two_sample_ttest_for_several_error_variance_values_and_mean_differences.csv')
test2_timings, test2_deviations, test2_results, test2_list_inputs, OhImAHugeHack = tex_table(file_path, TEST_2_FILENAME, ['Test02_V3_ConditionalPairedTTest.json'], 'Example_2_Power_results_for_a_Paired_Ttest.csv')
test3_timings, test3_deviations, test3_results, test3_list_inputs, OhImAHugeHack = tex_table(file_path, TEST_3_FILENAME, ['Test03_V3_ConditionalTwoSampleTTest3DPlot.json'], 'Example_3_Power_for_a_two_sample_ttest_for_various_sample_sizes_and_mean_differences.csv')
test4_timings, test4_deviations, test4_results, test4_list_inputs = tex_table_test_4(file_path,TEST_4_FILENAME, ['Example_4_Power_and_confidence_limits_for_a_univariate_model.json', 'Example_4_Power_and_confidence_limits_for_a_univariate_model_part2.json', 'Example_4_Power_and_confidence_limits_for_a_univariate_model_part3.json'], 'Example_4_Power_and_confidence_limits_for_a_univariate_model.csv')
test5_timings, test5_deviations, test5_results, test5_list_inputs = tex_table_test_5(file_path, TEST_5_FILENAME, ['Example_5_Power_for_a_test_of_interaction_in_a_multivariate_model.json'], 'Example_5_Power_for_a_test_of_interaction_in_a_multivariate_model.csv')
test6_timings, test6_deviations, test6_results, test6_list_inputs = tex_table_by_delta(file_path, TEST_6_FILENAME, ['Example_6_Power_and_confidence_limits_for_the_univariate_approach_to_repeated_measures_in_a_multivariate_model.json'], 'Example_6_Power_and_confidence_limits_for_the_univariate_approach_to_repeated_measures_in_a_multivariate_model.csv')
test7_timings, test7_deviations, test7_results, test7_list_inputs = tex_table_test7(file_path, TEST_7_FILENAME, ['Example_7_Power_for_a_time_by_treatment_interaction_using_orthogonal_polynomial_contrast_for_time.json'], 'Example_7_Power_for_a_time_by_treatment_interaction_using_orthogonal_polynomial_contrast_for_time.csv')

gaussian_test1_timings, gaussian_test1_deviations, gaussian_test1_results, gaussian_test1_list_inputs = tex_table_gaussian(file_path, GAUSSIAN_TEST_1_FILENAME, ['GLMM_F_g_Example_1_Median_power_for_the_Hotelling-Lawley_Trace_using_the_Satterthwaite_approximation_part_1.json', 'GLMM_F_g_Example_1_Median_power_for_the_Hotelling-Lawley_Trace_using_the_Satterthwaite_approximation_part_2.json', 'GLMM_F_g_Example_1_Median_power_for_the_Hotelling-Lawley_Trace_using_the_Satterthwaite_approximation_part_3.json'], 'Example_1_Median_power_for_the_HotellingLawley_Trace_using_the_Satterthwaite_approximation.csv')
gaussian_test4_timings, gaussian_test4_deviations, gaussian_test4_results, gaussian_test4_list_inputs = tex_table_gaussian(file_path, GAUSSIAN_TEST_4_FILENAME, ['GLMM_F_g_Example_4_Unconditional_power_for_the_Hotelling-Lawley_Trace_using_Davies_Algorithm_part_1.json', 'GLMM_F_g_Example_4_Unconditional_power_for_the_Hotelling-Lawley_Trace_using_Davies_Algorithm_part_2.json','GLMM_F_g_Example_4_Unconditional_power_for_the_Hotelling-Lawley_Trace_using_Davies_Algorithm_part_3.json'], 'Example_4_Unconditional_power_for_the_HotellingLawley_Trace_using_Davies_algorithm.csv')
gaussian_test5_timings, gaussian_test5_deviations, gaussian_test5_results, gaussian_test5_list_inputs = tex_table_gaussian(file_path, GAUSSIAN_TEST_5_FILENAME,['GLMM_F_g_Example_5_Median_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser-Greenhouse_and_Huynh-Feldt_tests_using_the_Satterthwaite_approximation_part_1.json', 'GLMM_F_g_Example_5_Median_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser_Greenhouse_and_Huynh-Feldt_tests_using_the_Satterthwaite_approximation_part_2.json', 'GLMM_F_g_Example_5_Median_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser_Greenhouse_and_Huynh-Feldt_tests_using_the_Satterthwaite_approximation_part_3.json'], 'Example_5_Median_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_GeisserGreenhouse_and_HuynhFeldt_tests_using_the_Satterthwaite_approximation.csv')
gaussian_test8_timings, gaussian_test8_deviations, gaussian_test8_results, gaussian_test8_list_inputs = tex_table_gaussian(file_path, GAUSSIAN_TEST_8_FILENAME, ['GLMM_F_g_Example_8_Unconditional_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser-Greenhouse_and_Huynh-Feldt_tests_using_Davies_Algorithm_part_1.json', 'GLMM_F_g_Example_8_Unconditional_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser-Greenhouse_and_Huynh-Feldt_tests_using_Davies_Algorithm_part_2.json', 'GLMM_F_g_Example_8_Unconditional_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser-Greenhouse_and_Huynh-Feldt_tests_using_Davies_Algorithm_part_3.json'], 'GLMM_F_g_Example_8_Unconditional_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser-Greenhouse_and_Huynh-Feldt_tests_using_Davies_algorithm.csv')

HOMEWORK_1_TITLE = """Exercise 1: Power for a Single Level Cluster Design"""
HOMEWORK_1_STUDY_DESIGN_DESCRIPTION ="""\\section{Short study description} 
A single level cluster-randomized trial of a drinking intervention in the workplace.
\\section{Study vignette}
The study was adapted from one described in Reynolds et al. (2015). Modifications may include changing clustering, treatment design, number of measures, outcomes, predictors, time spacing, and all inputs for the power or sample size analysis, including means, variances, standard deviations, sample sizes, powers, Type I error rates, correlations, covariates and correlations.

A single level cluster randomized study was planned to examine the efficacy of a workplace training program to reduce alcohol consumption. Researchers planned to randomize workplaces to one of two treatment groups. The entire workplace will receive the same treatment. A flow diagram for the study is shown in Exhibit 1.

The study will compare a workplace training program to a control treatment, in which there will be no training at all in the workplace. Although the Reynolds et al. (2015) study looked at drinking rates before and after treatment, in the proposed study, the outcome measure will be post-treatment drinking rate (drinks per occasion). Post-treatment drinking rate will be measured via interview 60 days after the treatment is completed. Workers will be asked both how many days they used alcohol and questions that allow quantifying their typical, per occasion quantity. The outcome (rate) will be calculated as the average number of drinks per day.

The independent sampling unit is the workplace. Within each workplace, the responses of the workers are correlated. This occurs because the workers talk together in the workplace, may choose to drink together, may compare drinking activities, and will undergo workplace training (or no training) together. The unit of randomization is the workplace. The unit of observation is the drinking rate for each worker.

It is expected that the measures of drinking rate for the different workers in the same work place will be correlated. Thus if participants Able and Baker both work at University Hospital, it is expected that their post-treatment drinking rates would be correlated. It is expected that the results for the different workplaces would be independent. If participant Charles works at Children's Hospital, the drinking rate for participant Charles should be independent of that of the rates of participants Able and Baker.

The between-independent sampling unit factor is treatment. The between independent sampling unit factor has two levels: a workplace training program and a control program. Workplace was the within-independent sampling unit factor in this study design.

The null hypothesis is that there will be no difference in post-treatment drinking rate between workers who received no training and workers who received the workplace program. The alternative hypothesis is that the training program will change post-treatment drinking rate. The researchers hope that the workplace training program will reduce drinking rate, making the average drinking rate smaller post-treatment in the treatment group relative to the control group. However, they would like to test for both larger and smaller post-treatment drinking rate. The researchers thought that there would be no change in drinking rate at all in the control group.

For the proposed study, every workplace is the same size, and has 15 workers. No other covariates are measured. There will be 20 workplaces assigned to each treatment program, for a total of 40 workplaces. From previous clinical experience, it is speculated that none of the workplaces will drop out of the study. In addition, previous experience suggests that none of the workers will drop out of the study. Thus, post-treatment drinking rate will measured on 600 people. Here, the number 600 is obtained by conducting the following calculation: 2 treatments‚20workplaces/treatment‚15workers/workplace =600.

From knowledge about the efficacy of the workplace intervention, and from previously published literature, the researchers speculate that the mean or average drinking rate for the control workplaces will be 1.24. The mean or average drinking rate for the workplaces where workers received treatment will be 0.73. The difference between 0.73 and 1.24 is considered to be of scientific interest. The common standard deviation of the measurement for each worker is expected to be 1.1.

The intracluster correlation coefficient will be 0.13. The intracluster is a number between -1 and 1 which represents the correlation between the post-treatment drinking rate of two workers within one cluster. Note that the correlation between the post-treatment drinking rates from workers from different workplaces is zero. This is because we assume that different workplaces are independent. In this study design, the workplace is the independent sampling unit.
Exhibit 1: A cluster randomized trial of post-treatment drinking rate.
      No training (Controls)
    Start
Randomize workplaces
Post-treatment drinking rate
Stop
Post-treatment drinking rate
 Workplace program

\\section{Statistical analysis plan}
Note: For your future reference, we describe three valid ways to analyze the data. All three are essentially equivalent. Please review the "Choosing the Test" lecture for details. For your write-up, you are welcome to select and describe only one analysis.

General linear multivariate model: We will fit a general linear multivariate model. The outcome will be drinking rate. There are 15 measures of drinking rate for each workplace. There are forty total workplaces, with twenty randomized to the workplace treatment program, and twenty randomized to the control group (no training). As predictors for the model, we will use an indicator variable which is one if there is workplace training program, and zero if there is no workplace training program.

We will test the difference between the average workplace drinking rates using the Hotelling- Lawley Trace test at a Type I error rate of 0.05.

The analysis assumes that the workplaces are independent, that the variance pattern for the residuals for each workplace is similar, that the results are finite, and that a linear model is a good fit for the data. The hypothesis test assumes that the residuals have a multivariate Gaussian distribution.

Two-sample t-test: We will form the average workplace drinking rates, by averaging the 15 worker drinking rates for each workplace. This will give us 40 averages, one for each workplace. Twenty averages will be for workplaces assigned to control, and twenty for workplaces assigned to treatment. We will conduct a two-sample t-test on the resulting sample averages, to test no difference between the treatments. We will use a two-sided Type I error rate of 0.05. We will use the Hotelling-Lawley trace statistic to assess the null hypothesis that
there is no difference in post-treatment drinking frequency between workers who received no training and workers who received the workplace program.

The assumption here is that the two treatment groups have equal variances, and that the sample size is large enough that the test statistic has an approximate t distribution.

Mixed model: We will fit a general linear mixed model. The outcome variable will be the post- treatment drinking rate. As predictors, we will use two indicator variables. The first indicator is one if there is workplace training program, and zero if there is no workplace training program, and the workplace is in the control group. The second indicator is one if the workplace is in the control group, and one if it is in the workplace training program. To account for correlation within the workplace, we will fit a random effect for workplace. Doing so produces a compound symmetric error variance matrix.
\\section{Inputs for power and sample size analysis}
The goal of this analysis is to calculate power, for a given sample size. For this power analysis, we need several inputs.

1. Type 1 error: We set α œ !Þ!&Þ
2. Sample size: There are 40 workplaces, 20 assigned to each treatment. Notice that the design is balanced, which means that there are an equal number of workplaces in each treatment group. Recall that the sample size is the number of independent sampling units, not the number of units of observations.
3. Cluster size: There are 15 people in each workplace.
4. Intracluster correlation: The intraclass correlation is 0.13.
5. Standard deviation: The standard deviation of the post-treatment drinking rate is 1.1.
6. Treatment means: The mean for the control workplaces was 1.24, and the mean for treated workplaces was 0.73.
7. Scale factors: The scale factor to be used for means is 1. The scale factor to be used for variability is 1.

References cited

Reynolds GS, Bennett JB. A cluster randomized trial of alcohol prevention in small businesses: a cascade model of help seeking and risk reduction. Am J Health Promot. 2015;29(3):182– 191. (Use difference score as the outcome)"""
HOMEWORK_2_TITLE = """Exercise 2: Sample Size Analysis for a Longitudinal Study"""
HOMEWORK_2_STUDY_DESIGN_DESCRIPTION ="""\\section{Short study description}
A longitudinal study with both within- and between-independent sampling unit factors.
\\section{Study vignette}
This study is a hypothetical replication of the one described in Logan et al., 1995. The study flow diagram is shown in Exhibit 1. Modifications may include changing clustering, treatment design, number of measures, outcomes, predictors, time spacing, and all inputs for the power or sample size analysis, including means, variances, standard deviations, sample sizes, powers, Type I error rates, correlations, covariates and correlations.

Exhibit 1: A longitudinal randomized controlled clinical trial of a sensory focus intervention on memory of pain.
 
    Sensory Focus
 Randomize Participant
Month 0 Data
Month 0 Data
Month 6 Data
Month 6 Data
Month 12 Data
Stop
Month 12 Data
     Standard of Care

Researchers plan to conduct a longitudinal randomized controlled clinical trial in patients who had experienced a root canal. The outcome of interest is the memory of pain. The goal of the study is to determine if dental patients who were instructed to use a sensory focus have a different pattern of long-term memory of pain than participants who did not. Researchers hypothesize that the pattern of memory of pain would be different for those who had the intervention, and those who were in the control group.

The null hypothesis is that the pattern of memory of pain over time would be no different between those who had the intervention, and those who were in the control group. The alternative hypothesis is that the pattern of memory of pain over time would be different for the control group and the intervention group. This is an interaction hypothesis, also known as a between-by-within hypothesis. A picture of an interaction effect is shown in Exhibit 2.

Exhibit 2: A graph of the possible outcomes over time for the memory of pain trial. The pattern of outcomes over time differs between the two intervention groups, a pattern consistent with time-by-treatment interaction.

Participants are to be selected and randomly assigned to either the sensory focus intervention or the standard-of-care intervention. An equal number of patients will be assigned to each treatment group. Patients in the intervention group will listen to automated audio instructions to pay close attention only to the physical sensations in their mouth. Patients in the standard-of-care group will listen to automated audio instruction on a neutral topic to control for media and attention effects.

All patients will be queried three times about their memory of pain. They will be asked to describe their memory of pain immediately, at six months, and at twelve months after the root canal and intervention.

In this study, the outcome measure is the memory of pain. The independent sampling unit is the patient. The unit of randomization is the patient. The unit of observation is the memory of pain at each time point. It is expected that the three longitudinal measures over time for each patient will be correlated. It is also expected that each study participant will be independent from other study participants. The between-independent sampling unit factor is treatment. Treatment has two levels: sensory focus intervention and control treatment. The within- independent sampling unit factor is time. Time has three levels: 0 months, 6 months and 12 months. It is expected that repeated measurements within each person will be correlated.

Gedney, Logan, and Baron (2003) identified predictors of the amount of experienced pain recalled over time. One of the findings was that memory of pain intensity at 1 week and 18 months had a correlation of 0.4.

Given the previous research, for this exercise we assume that the correlation between measures 6 months apart will be 0.5. Also we assume that the correlation between measures 12 months apart will be 0.4.

Logan, Baron, and Kohout (1995) examined whether sensory focus therapy during a root canal procedure could reduce a patient's experienced pain. The investigators assessed experienced pain on a 5 point scale both immediately and at one week following the procedure. The standard deviation of the measurements was 0.9.

Based on clinical expertise, the investigators speculate that the pattern of means for the two groups will be as shown in Exhibit 3.

Exhibit 3: Predicted mean outcome for memory of pain score by treatment and time.

The goal is to calculate a reasonable sample size for the study. The investigators would like to know what the sample size should be for power values of 0.85, 0.90 and 0.95.
\\section{Statistical analysis plan}

Note: For your future reference, we describe two valid ways to analyze the data. Both are roughly equivalent. Please review the "Choosing the Test" lecture for details. For your write-up, you are welcome to select and describe only one analysis.

General linear multivariate model: We will fit a general linear multivariate model. The outcome variables will be the three repeated measurements of memory of pain. The predictors will be two indicator variables, which, respectively, take on the value 1 if the person was assigned to sensory focus, and 0 otherwise, and take on the value 1 if the person was assigned to standard-of-care, and 0 otherwise. We will use a Hotelling-Lawley trace statistic to assess the null hypothesis that the pattern of memory of pain over time is no different between those who had the intervention, and those who were in the control group. We will use a Type I error rate of 0.05. This modeling technique assumes no missing data for any person for any of the repeated measurements, and assumes equal error variance, independence of the independent sampling units, finite second moments, and linearity, which means that the outcome could be described as a linear function of the predictors. We will use regression diagnostics and jackknifed studentized residuals to examine the assumptions.

General linear mixed model: We will fit a general linear mixed model. The outcome variables will be the three repeated measurements of memory of pain. The predictors will be two indicator variables, which, respectively, take on the value 1 if the person was assigned to sensory focus, and 0 otherwise, and take on the value 1 if the person was assigned to standard- of-care, and 0 otherwise. We will use a Wald statistic with Kenward-Roger degrees of freedom to assess the null hypothesis that the pattern of memory of pain over time is no different between those who had the intervention, and those who were in the control group.

We will use an unstructured covariance matrix, and assume that the variance-covariance matrix of the errors is the same for each person. We will use a Type I error rate of 0.05.

This modeling technique assumes no missing data for any person for any of the repeated measurements, and assumes equal error variance, independence of the independent sampling units, finite second moments, and linearity, which means that the outcome could be described as a linear function of the predictors. We will use regression diagnostics and jackknifed studentized residuals to examine the assumptions.

    Baseline 6 months Sensory Focus 3.6 2.8 Standard of Care 4.5 4.3

12 months 0.9 3.0

\\section{Inputs for sample size analysis}

The goal of this analysis is to calculate sample size, for a given power. For this sample size analysis, we need several inputs.

1. Type 1 error: We set α œ !Þ!&Þ
2. Power: We consider power values of 0.85, 0.90 and 0.95. The investigators would like to
find the three different sample size values associated with the three power values.
3. Cluster size: There are no clusters.
4. Randomization plan: We plan to have equal numbers of people randomized to the sensory focus treatment, and the standard of care treatment.
5. Number of repeated measurements: There are 3 repeated measures over time.
6. Pattern of means. The pattern of means is shown in Exhibit 3, repeated here for convenience.

Exhibit 3: Predicted mean outcome for memory of pain by treatment and time.

7. Correlation: We assume that the correlation between measures 6 months apart will be 0.5 and that the correlation between measures 12 months apart will be 0.4.
8. Standard deviation: The standard deviation of the memory of pain score is 0.9 across all repeated measurements.
9. Scale factors: The scale factor to be used for means is 1. The scale factor to be used for variability is 1.

References cited
Logan, H. L., R. S. Baron, and F. Kohout. “Sensory Focus as Therapeutic Treatments for Acute Pain.” (1995) Psychosomatic Medicine 57(5): 475–84."""
HOMEWORK_3_TITLE = """Exercise 3: Power for a Multilevel Study"""
HOMEWORK_3_STUDY_DESIGN_DESCRIPTION ="""\\section{Study name}

A multilevel study with a hypothesis test of a between-independent sampling unit factor.
\\section{Study vignette}

The study described in this homework exercise is a strongly modified version of the one described in Piquette et al. (2014).

Researchers plan to conduct a randomized controlled clinical trial of an intervention designed to help young children learn fundamental early literacy skills. The intervention is named ABRACADABRA, which is an acronym for "A Balanced Reading Approach for Canadians." Alternative approaches include two standardized Canadian training options for teaching literacy. The first is an English Language Arts program, and the second is a bi-lingual program for native English speakers designed to build literacy in both French and English.

Researchers plan to randomize 45 schools, with 15 in each treatment arm. That is, 15 schools will get ABRACADABRA, 15 schools will get the English Language Arts Program, and 15 schools will get a bi-lingual French/English program.

All of the schools are very far apart from each other. In fact, they are so far apart that the students, parents and teachers in each school have no contact with the students, parents and teachers in any other schools. Thus, schools can be considered to be independent.

Initial study planning will begin by assuming each school has exactly 4 kindergarten classrooms, each with 5 students, all of whom will take part in the trial. Thus, each school has 20 total students, calculated as 20 œ (5 students/classroom) ‚ (4 classrooms/school). In most real scenarios, there will be a different number of students in each classroom. There will be a different number of classrooms in each school. There will be a different number of schools in each neighborhood. For scenarios with different numbers in each cluster, we will discuss how to handle power and sample size analysis later.

The extent that gender, age and ethnicity of the students is related to the outcomes is not going to be considered for the initial study planning.

The intraclass correlation coefficient for classrooms within schools was assumed to be 0.11. The intraclass correlation coefficient for students within classrooms was assumed to be 0.04.

Students will be evaluated on three composite outcome scores that are considered to have a multivariate normal distribution. The three composite scores are letter-sound knowledge (LSK), the Comprehensive Test of Phonological Processing (CTOPP) blending words subtest age equivalent scores, and the Group Reading Assessment and Diagnostic Evaluation (GRADE) listening comprehension subtest stanine score.

Students will be measured before and after the intervention, and for each component, a difference score will be calculated as postpre. The outcomes of interest are the difference scores for the three composite scores. Each student will contribute three difference scores: one for LSK, one for CTOPP and one for GRADE. The common standard deviation (across the three treatment arms) for GRADE is 4.4, for LSK is 4.2, and for CTOPP is 0.6.

Exhibit 1: Predicted mean (standard deviation) difference scores for three literacy scale scores, stratified by treatment arm.
 
ABRA English

Language Arts

bi-lingual literacy Program

GRADE LSK CTOPP 0.3 0.3 0.3

0.1 0.1 0.1 0.1 0.1 0.1

Initial study planning will begin by assuming no missing data, which corresponds to requiring everyone present for either the pre or post test is present for both. Subsequent refinement of the sample size analysis could include an allowance for missing data, if it is deemed appropriate.
Based on knowledge of previous studies, scientists have a good guess as to what the correlation matrix looks like for the difference scores of the three outcome measures.

Exhibit 2: Correlation Matrix for difference scores

The three difference variables define a multivariate response profile. Scientists hypothesize that the three literacy training programs do not differ in any combination of the outcome differences. The scientists wonder what their power will be for the proposed trial.
\\section{Statistical analysis plan}

Scientists plan to fit a general linear multivariate model with the 20 difference scores in LSK, CTOPP and GRADE, respectively, observed per school as outcomes. There will be 45 schools contributing data to the model, with 15 assigned to each of three treatment arms. As predictors, the scientists plan to use indicator variables for the three treatments. The scientists plan to test a multivariate analysis of variance (MANOVA) hypothesis. Scientists will use the Hotelling- Lawley trace statistic at a Type I error rate of 0.05 to evaluate the null hypothesis of no differences in response difference profiles among the three literacy programs. The scale factor to be used for means is 1. The scale factor to be used for variability is 1.

References cited

Piquette, Noella A., Robert S. Savage, and Philip C. Abrami. “A Cluster Randomized Control Field Trial of the ABRACADABRA Web-Based Reading Technology: Replication and Extension of Basic Findings.” (2014) Frontiers in Psychology, 5. doi:10.3389/fpsyg.2014.01413."""

HOMEWORK_4_TITLE = """Exercise 4: Sample Size Analysis for a Multilevel Study with Longitudinal Repeated Measures"""
HOMEWORK_4_STUDY_DESIGN_DESCRIPTION ="""\\section{Short Study Description}

A multilevel study with longitudinal repeated measures.

\\section{Study vignette}

The study described in this homework exercise is a strongly modified version of the one described in Komro et al. (2008). Modifications may include changing clustering, treatment design, number of measures, outcomes, predictors, time spacing, and all inputs for the power or sample size analysis, including means, standard deviations, sample sizes, powers, Type I error rates, correlations and covariates.

Scientists want to find a sample size for a planned randomized controlled clinical trial. They are interested in power of at least 0.90.

Researchers plan to conduct a randomized controlled clinical trial of an intervention designed to reduce adolescent alcohol use. The goal was to compare the intervention with no intervention. After obtaining consent from students, parents, teachers, and administrative staffs, researchers grouped schools within neighborhoods to form neighborhood groups. Neighborhood groups were randomized to either the alcohol use intervention or standard of care, using a 2 to 1 randomization scheme. The 2 to 1 randomization scheme has two times the number of neighborhoods randomized to the alcohol education program, as a result of community demand. For logistic and cost reasons, as well as the desire to ensure the smallest group is of sufficient size, the scientists wish to restrict total sample size to the range 30-45 neighborhoods.

Each student will be surveyed at baseline and again three more times after the treatment begins. Thus there will be four measurements, at baseline, in the spring of sixth grade, the spring of seventh grade, and the spring of eighth grade. The survey will use a detailed diet recall method to obtain an alcohol use scale for each student at each time point. Previous similar modeling with this variable has produced acceptably normally distributed jackknifed studentized residuals without transformation of the data.

For the purposes of this question, we can assume that there were an equal number of students in each classroom, an equal number of classrooms per school, and equal number of schools per neighborhood group.

We expect all neighborhoods, schools and classrooms to stay with the study throughout the entire time. However, we know that students move in and out of the district. Previous studies of student absences has reassured us that the absences of the students are not related to the neighborhood, classroom or school, nor to the use or non-use of the alcohol treatment program, nor to the age of the student.

There are 20 students in each classroom, with intracluster correlation of 0.09. There are 3 classrooms per school, with an intracluster correlation coefficient of 0.04. There are 2 schools per neighborhood, with an intracluster correlation coefficient of 0.03.

The researchers believe, from previous experience with similar data, that the correlation of scores across time followed a LEAR model with base correlation of 0.6 and a decay rate of 0.7, leading to a decrease in correlation of about 0.10 per unit time. The LEAR model (Simpson et al., 2010) allows correlation to be strong at first, and then die off with time at a rate controlled by the decay parameter. The AR(1) model is a special case.

Measurements will be conducted at 0, 6, 18 and 30 months. For the purposes of using GLIMMPSE, describe time values as 0, 1, 3 and 5 HalfYears. The scaling of the

time values changes the LEAR correlation matrix due to the parameter structure chosen in GLIMMPSE.

Again from previous experience and work, the researchers are interested in the pattern of means in the following table.

Exhibit 1: Predicted neighborhood mean alcohol use scores stratified by treatment arm.

The common standard deviation is 4.

\\section{Statistical analysis plan}

Scientists plan to fit a general linear mixed model with the alcohol use scores for each student as the outcomes. As predictors, they will use indicator variables for the two treatments, the alcohol education program and the standard of care. The scientists plan to account for correlation of schools within neighborhood groups, classrooms within schools, and students within classrooms. In all three levels, the schools are assumed to exchangeable within neighborhoods, the classrooms within schools, and the students within classrooms, leading to compound symmetry for each level of clustering, and a direct-product structure. The longitudinal repeated measures of alcohol use over time will assume a LEAR covariance structure (Simpson et al., 2010).

Scientists plan to use a Wald statistic with Kenward-Roger degrees of freedom (which corresponds to a Hotelling-Lawley Test for complete data), and a Type I error rate of 0.05 to evaluate the null hypothesis of no difference in pattern of average alcohol use scores over time between the treatments. The scale factor to be used for means is 1. The scale factor to be used for variability is 1.

References cited
Komro, Kelli A., Cheryl L. Perry, Sara Veblen-Mortenson, Kian Farbakhsh, Traci L. Toomey, Melissa H. Stigler, Rhonda Jones-Webb, Kari C. Kugler, Keryn E. Pasch, and Carolyn L. Williams. “Outcomes from a Randomized Controlled Trial of a Multi- Component Alcohol Use Preventive Intervention for Urban Youth: Project Northland Chicago.” Addiction (Abingdon, England) 103, no. 4 (April 2008): 606–18. doi:10.1111/j.1360-0443.2007.02110.x.

Simpson, S. L., Edwards, L. J., Muller, K. E., Sen, P. K., & Styner, M. A. (2010). A linear exponent AR(1) family of correlation structures. Statistics in Medicine, 29(17), 1825– 1838. http://doi.org/10.1002/sim.3928"""

HOMEWORK_5_TITLE = """Exercise 5: Sample Size Analysis for a Planned Subgroup Analysis"""
HOMEWORK_5_STUDY_DESIGN_DESCRIPTION ="""\\section{Short study description}
A multivariate study with between-independent sampling unit factors.
\\section{Study vignette}
The study described is a possible future extension of the study conducted by Bullitt et al. (2005). We have made strong efforts in this vignette to hew as much as possible to the science behind the study, and to include, as much as possible, reasonable values. However, we did create some values in order to set up a reasonable sample size analysis. We tried to indicate where we used real values, and where we used speculated values. In addition, we have made up some details of the proposed studies, such as the four major genotypes of VegF that are considered in the proposed study, and the chemotherapeutic regimen described that affects vessel tortuosity.

Before you read this study vignette, please read the accompanying article by Bullitt et al. (2005) carefully. The goal is not necessarily to understand the science. The goal is to look carefully at the article to find where the authors published means and standard deviations.

For the Muller et al. (2007) article, please look only at Equation 6 on page 3648, which displays a covariance matrix. In the class, we talked about correlation matrices. Recall that a correlation matrix describes the associations between two or more variables. A covariance matrix is like a correlation matrix in that it describes the associations between two or more variables. However, a correlation matrix contains scaled numbers, between -1 and 1. A covariance matrix contains unscaled numbers, still in the scale of the original variables.

In the class, we stressed the importance of publishing correlation or covariance values, or both, for the future use of researchers in the field who are conducting power or sample size analyses. We included the Muller et al. (2007) paper to show that some authors do publish these values. However, we note that the values are not published in the Bullitt et al. (2005) manuscript. Unfortunately, publication of correlation values, or covariance values, or both, is unusual.

In Bullitt et al. (2005) manuscript, the researchers described a measure of vessel tortuosity in the brain. Tortuosity is a measure of how twisted a vessel is. Tumors develop new blood vessels as they grow. Frequently, the new blood vessels have many small bends in them. Measuring tortuosity via MRA can help scientists differentiate between normal tissue and tumor tissue, and measure the effectiveness of treatments at stopping tumor growth and progression.

Bullitt et al. (2005) considered multiple ways to quantify tortuosity. This exercise will concentrate on a single way to quantify tortuosity, called SOAM1. In the Bullitt study, scientists quantified vessel tortuosity using SOAM1 in four regions of the brain. Two of the regions, the left and right middle cerebral groups, had similar summary SOAM1 measurements, as shown in Figure 2, on page 45, and in Table 4, page 46.

In the proposed study, the scientists planned to use SOAM1 as the outcome. They planned to measure this outcome in two regions of the brain for every patient, the left middle cerebral group, and the right middle cerebral group. This means that there are two repeated measurements of SOAM1 for each patient.

Using the information from Table 1, page 45 in Bullitt et al. (2005), please fill in the following table for use in your power analysis.

Exhibit 1: Statistics for summary SOAM1 measures by brain regions.
    Left Middle Cerebral Group Right Middle Cerebral Group

Mean Standard Deviation
 
The Bullitt et al. (2005) paper does not provide correlations between measurements of SOAM1 on different brain regions. Converting the covariance matrix that appears in Equation 6 on page 3648 into a correlation matrix yields the correlation matrix shown in the following table.

Exhibit 2: Correlation between summary SOAM1 measures for different brain regions

Bullitt et al. (2005) studied the SOAM1 variable, and concluded that the distribution was appropriately normal, or Gaussian. You can look at their results by examining the p-values that appear in Table 1.

Bullitt et al. (2005) had theorized that abnormal tortuosity in vessels was perhaps caused by increases in nitrous oxide induced by VegF. There are four major genotypes of VegF, designated, for convenience, Genotype A, Genotype B, Genotype C and Genotype D.

A new group of investigators was interested in studying the tortuosity response of blood vessels in the left and right middle cerebral groups. The plan was to recruit study participants with glioblastoma multiforme, a brain tumor. It was expected that the researchers would be able to recruit equal numbers of those with Genotypes A, B and C. Because of the frequency of Genotype D in the population, there would be roughly twice as many study participants with Genotype D as with Genotype A.

Exhibit 3: Relative sizes of genotype groups in study population.

The researchers planned to randomize the study population either to a placebo, or to a new chemotherapeutic regimen. They wanted to measure vessel tortuosity in the left and right middle cerebral groups as the outcomes of the study. That is, for each study participant, they would have two repeated measurements of SOAM1, one on the left cerebral middle group, and the other on the right cerebral middle group. To ensure equal allocation within each group, they planned a block randomization scheme with study participants randomized one-to-one to treatment or placebo within each genotype group.

The researchers were confident that the response of each individual to treatment would be independent, even if the individuals went to the same clinic. They thought that the responses of the two brain regions for each study participant would be correlated.

The researchers wanted to test to see if there would be an interaction between treatment and genotype on the average response to treatment across the two brain regions. Recall that an interaction hypothesis describes how two factors (here, the two between ISU factors, treatment and genotype) interact to change the response. This is the same as asking whether the effect of treatment differs across the genotypes, where we are measuring the effect of treatment by looking at the average across the two brain regions.

    Left Right Middle Middle Cerebral Cerebral

Left Middle Cerebral Group 1 0.53 Right Middle Cerebral Group 0.53 1

     Size Ratio Genotype A 1

Genotype B 1 Genotype C 1 Genotype D 2

In GLIMMPSE, when one requests the treatment by genotype interaction, GLIMMPSE automatically assumes that one wishes to average across the two brain regions, and calculates the power or sample size in that manner. This is exactly what the researchers wanted to do.

By the way, the variance of an average decreases with the number of observations that one averages. Remember that power goes up as variance goes down. Thus, you will get different answers for the power and sample size calculation if you assume that SOAM1 is measured once, twice, or four times. You can try this out empirically by changing the number of repeated measurements of SOAM1. Because of this, it is important to describe the study as having one outcome variable, with two repeated measurements.

They theorized that the pattern of mean responses would be as shown in the following table.

Exhibit 4: Predicted responses by treatment and genotype.
    Treatment Genotype
Placebo A Placebo B Placebo C Placebo D Chemotherapy A Chemotherapy B Chemotherapy C Chemotherapy D
Mean Left Middle Cerebral SOAM1
3.12 3.12 3.12 3.12 2.1 2.3 2.5 2.7
Mean Right Middle Cerebral SOAM1
3.17 3.17 3.17 3.17 2.15 2.35 2.55 3.25

 Glioblastoma multiforme is an almost uniformly fatal disease with a rapid disease course. The chemotherapeutic regimen involved targeted mRNA, and previous studies had shown a low rate of side effects. The MRA imaging regimen used to measure study outcomes does have a low risk of anaphylaxis due to the use of a gadolinium contrast agent, but, weighing risks and benefits, the Institutional Review Board thought that the risk was acceptable, given the risks of the disease.

After an ethics consultation, the researchers decided that it would be preferable to have the study large enough to test the hypothesis, rather than adopting a conservative approach and limiting sample size to limit exposure to the risks of the study. Thus, they planned to use the larger of the two variance estimates they had. Notice that the researchers had a choice of two standard deviation values, as shown in Exhibit 1. To be conservative, round the standard deviation you choose to one value behind the decimal point.

Roughly 180 patients meeting the eligibility criteria are seen each month by the large glioblastoma clinic at the high volume tertiary care clinic at which the study will be done. Roughly 20% of the patients will consent to the study. The chance of consent is not associated with genotype, nor with response to treatment, nor with baseline tortuosity. The study investigators think that almost 30% of the participants will be lost to follow-up and not complete the study, for reasons independent of disease severity, genotype, or response. From previous experience, the investigators believe that if people complete the study, they are quite likely to be able to measure the vessel tortuosity in both brain regions. It is unlikely that they will be able to measure response in only one region, or in no regions at all.

Feasible sample sizes for the study would be 30, 40 and 50, with the smallest group size, respectively of 3, 4 and 5. The goal is to figure out what the power is for each of the sample sizes, and to choose the smallest sample size such that the power is at least 0.95.

\\section{Statistical analysis plan}

We will fit a general linear mixed model. The outcome variables will be the two repeated measurements of SOAM1, one in the left cerebral middle group, and one in the right cerebral middle group. The predictors will be eight indicator variables for the genotype by treatment groups. Each indicator variable will take on the value 1 if the study participant is a member of the specific genotype-by-treatment group, and 0 otherwise. We will use a Wald statistic with Kenward-Roger degrees of freedom (which corresponds to a Hotelling-Lawley test for complete data) to assess the null hypothesis that there is no interaction between genotype and treatment on the average response over the two brain regions. We will use an unstructured covariance matrix, and assume that the variance-covariance matrix of the errors is the same for each person. We will use a Type I error rate of 0.05. The scale factor to be used for means is 1. The scale factor to be used for variability is 1.

This modeling technique assumes equal error variance, independence of the independent sampling units, finite second moments, and linearity, which means that the outcome could be described as a linear function of the predictors. We will use regression diagnostics and jackknifed studentized residuals to examine the assumptions.

References cited

Bullitt, E., Muller, K. E., Jung, I., Lin, W., & Aylward, S. (2005). Analyzing attributes of vessel populations. Medical Image Analysis, 9(1), 39–49. http://doi.org/10.1016/j.media.2004.06.024
Muller, K. E., Edwards, L. J., Simpson, S. L., & Taylor, D. J. (2007). Statistical tests with accurate size and power for balanced linear mixed models. Statistics in Medicine, 26(19), 3639– 3660. http://doi.org/10.1002/sim.2827
Muller, K. E., & Stewart, P. W. (2006). Linear Model Theory: Univariate, Multivariate, and Mixed Models (1 edition). Hoboken, N.J: Wiley-Interscience."""

TEST_1_TITLE = """GLMM(F) Example 1. Power for a two sample t-test for several error variance values and mean differences"""
TEST_1_STUDY_DESIGN_DESCRIPTION ="""The study design for Example 1 is a balanced, two-group design.
We calculate power for a two-sample t-test comparing the mean responses between the two groups.
The example is based on the results in  Muller, K. E., \\& Benignus, V. A. (1992). \\emph{Neurotoxicology and teratology}, \\emph{14}(3), 211-219."""

TEST_2_TITLE = """GLMM(F) Example 2. Power results for a Paired T-test"""
TEST_2_STUDY_DESIGN_DESCRIPTION = """The study design in Example 2 is a one sample design with a pre- and post-measurement for each participant.
We calculate power for a paired t-test comparing the mean responses at the pre- and post-measurements. We
express the paired t-test as a general linear hypothesis in a multivariate linear model."""

TEST_3_TITLE="""GLMM(F) Example 3. Power for a two sample t-test for various sample sizes and mean differences"""
TEST_3_STUDY_DESIGN_DESCRIPTION="""The study design for Example 3 is a balanced, two sample design with a single response variable. We calculate
power for a two-sample t-test comparing the mean responses between the two independent groups. The example
demonstrates changes in power with different sample sizes and mean differences."""

TEST_4_TITLE="""GLMM(F) Example 4. Power and con dence limits for a univariate
model"""
TEST_4_STUDY_DESIGN_DESCRIPTION="""The study design for Example 4 is a balanced two group design. We calculate power for a two-sample t-test
comparing the mean response between the groups. We calculate con dence limits for the power values. The
example is based on Figure 1 from
Taylor, D. J., & Muller, K. E. (1995). Computing Con dence Bounds for Power and Sample Size of the General
Linear Univariate Model. The American Statistician, 49(1), 43-47."""

TEST_5_TITLE="""GLMM(F) Example 5. Power for a test of interaction in a multivari-
ate model"""
TEST_5_STUDY_DESIGN_DESCRIPTION="""The study design for Example 5 is a balanced four-sample designwith three repeated measures over time.
We calculate power for a test of the group by time interaction. The unstructured covariance model is most
appropriate for the design. The example demonstrates the di erence in power depending on the choice of
statistical test when assumptions of sphericity are unlikely to hold."""

TEST_6_TITLE="""GLMM(F) Example 6. Power and condence limits for the univariate approach to repeated measures in a multivariate model"""
TEST_6_STUDY_DESIGN_DESCRIPTION="""The study design for Example 6 is a factorial design with two between participant factors and one within participant factor. Participants were categorized by gender and classied into ve age groups. For each participant,
cerebral vessel tortuosity was measured in four regions of the brain. We calculate power for a test of the gender
by region interaction. Confidence limits are computed for the power values.
The matrix inputs below show the starting point for the B matrix. The third column of the matrix (i.e. vessel
tortuosity in the third region the brain) is modied throughout the validation experiment to progressively increase
the eect of gender. Mean values for males are increased by 0.0008 at each iteration, while corresponding values
for females are decremented by 0.0008. The process is restarted for each statistical test. For example, the first
power calculated for a given test would use the following B matrix 

\\mathbf{B} = \\begin{bmatrix}
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
\\end{bmatrix} + \\begin{bmatrix}
0 & 0 & 0.0008 & 0 \\\\
0 & 0 & 0.0008 & 0 \\\\
0 & 0 & 0.0008 & 0 \\\\
0 & 0 & 0.0008 & 0 \\\\
0 & 0 & 0.0008 & 0 \\\\
0 & 0 & -0.0008 & 0 \\\\
0 & 0 & -0.0008 & 0 \\\\
0 & 0 & -0.0008 & 0 \\\\
0 & 0 & -0.0008 & 0 \\\\
0 & 0 & -0.0008 & 0 \\\\
\\end{bmatrix}

= \\begin{bmatrix}
2.9 & 3.2 & 3.5008 & 3.2 \\\\
2.9 & 3.2 & 3.5008 & 3.2 \\\\
2.9 & 3.2 & 3.5008 & 3.2 \\\\
2.9 & 3.2 & 3.5008 & 3.2 \\\\
2.9 & 3.2 & 3.5008 & 3.2 \\\\
2.9 & 3.2 & 3.4992 & 3.2 \\\\
2.9 & 3.2 & 3.4992 & 3.2 \\\\
2.9 & 3.2 & 3.4992 & 3.2 \\\\
2.9 & 3.2 & 3.4992 & 3.2 \\\\
2.9 & 3.2 & 3.4992 & 3.2 \\\\
\\end{bmatrix}

And the last power calculated for a given test would use the following B matrix 

\\mathbf{B} = \\begin{bmatrix}
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
\\end{bmatrix} + \\begin{bmatrix}
0 & 0 & 0.2 & 0 \\\\
0 & 0 & 0.2 & 0 \\\\
0 & 0 & 0.2 & 0 \\\\
0 & 0 & 0.2 & 0 \\\\
0 & 0 & 0.2 & 0 \\\\
0 & 0 & -0.2 & 0 \\\\
0 & 0 & -0.2 & 0 \\\\
0 & 0 & -0.2 & 0 \\\\
0 & 0 & -0.2 & 0 \\\\
0 & 0 & -0.2 & 0 \\\\
\\end{bmatrix}

= \\begin{bmatrix}
2.9 & 3.2 & 3.7 & 3.2 \\\\
2.9 & 3.2 & 3.7 & 3.2 \\\\
2.9 & 3.2 & 3.7 & 3.2 \\\\
2.9 & 3.2 & 3.7 & 3.2 \\\\
2.9 & 3.2 & 3.7 & 3.2 \\\\
2.9 & 3.2 & 3.3 & 3.2 \\\\
2.9 & 3.2 & 3.3 & 3.2 \\\\
2.9 & 3.2 & 3.3 & 3.2 \\\\
2.9 & 3.2 & 3.3 & 3.2 \\\\
2.9 & 3.2 & 3.3 & 3.2 \\\\
\\end{bmatrix}


This is based on an example presented in
Gurka, M. J., Coey, C. S., & Muller, K. E. (2007). Internal pilots for a class of linear mixed models with
Gaussian and compound symmetric data. Statistics in Medicine, 26(22), 4083-4099.
"""

TEST_7_TITLE="""Example 7. Power for a time by treatment interaction
using orthogonal polynomial contrast for time"""
TEST_7_STUDY_DESIGN_DESCRIPTION="""The study design for Example 7 is a balanced two sample design with  ve repeated measures over time. We
calculate power for a test of the time trend by treatment interaction. The example demonstrates the use of an
orthogonal polynomial contrast for the e ect of time."""

GAUSSIAN_TEST_1_TITLE="""GLMM(F, g) Example 1. Median power for the Hotelling-Lawley
Trace, using the Satterthwaite approximation"""
GAUSSIAN_TEST_1_STUDY_DESIGN_DESCRIPTION="""The study design in Example 1 is a three sample design with a baseline covariate and four repeated measurements.
We calculate the median power for a test of no di erence between groups at each time point, using the Hotelling-
Lawley Trace test. A Satterthwaite approximation is used to obtain the approximate distribution of the test
statistic under the alternative hypothesis. Median power is calculated for the following combinations of mean
di erences and per group sample sizes.


1. Per group sample size of 5, with beta scale values 0.4997025, 0.8075886, and 1.097641
2. Per group sample size of 25, with beta scale values 0.1651525, 0.2623301, and 0.3508015
3. Per group sample size of 50, with beta scale values 0.1141548, 0.1812892, and 0.2423835


The example is based on Table II from
Glueck, D. H., & Muller, K. E. (2003). Adjusting power for a baseline covariate in linear models. Statistics in
Medicine, 22(16), 2535-2551."""

GAUSSIAN_TEST_4_TITLE="""GLMM(F, g) Example 4. Unconditional power for the Hotelling-
Lawley Trace, using Davies algorithm"""
GAUSSIAN_TEST_4_STUDY_DESIGN_DESCRIPTION="""The study design in Example 4 is a three sample design with a baseline covariate and four repeated measurements.
We calculate the unconditional power for a test of no di erence between groups at each time point, using the
Hotelling-Lawley Trace test. The exact distribution of the test statistic under the alternative hypothesis is
obtained using Davies' algorithm described in
Davies, R. B. (1980). Algorithm AS 155: The Distribution of a Linear Combination of Chi-Square Random
Variables. Applied Statistics, 29(3), 323-333.
Unconditional power is calculated for the following combinations of mean di erences and per group sample sizes.

1. Per group sample size of 5, with beta scale values 0.4997025, 0.8075886, and 1.097641
2. Per group sample size of 25, with beta scale values 0.1651525, 0.2623301, and 0.3508015
3. Per group sample size of 50, with beta scale values 0.1141548, 0.1812892, and 0.2423835

The example is based on Table II from
Glueck, D. H., & Muller, K. E. (2003). Adjusting power for a baseline covariate in linear models. Statistics in
Medicine, 22(16), 2535-2551."""

GAUSSIAN_TEST_5_TITLE="""GLMM(F, g) Example 5. Median power for the uncorrected univari-
ate approach to repeated measures, Box, Geisser-Greenhouse, and
Huynh-Feldt tests, using the Satterthwaite approximation"""
GAUSSIAN_TEST_5_STUDY_DESIGN_DESCRIPTION="""The study design in Example 5 is a three sample design with a baseline covariate and four repeated measurements.
We calculate the median power for a test of no di erence between groups at each time point. We calculate
median power for the uncorrected univariate approach to repeated measures, Box, Geisser-Greenhouse, and
Huynh-Feldt tests.A Satterthwaite approximation is used to obtain the approximate distribution of the test
statistic under the alternative hypothesis. Median power is calculated for the following combinations of mean
di erences and per group sample sizes.

1. Per group sample size of 5, with beta scale values 0.4997025, 0.8075886, and 1.097641
2. Per group sample size of 25, with beta scale values 0.1651525, 0.2623301, and 0.3508015
3. Per group sample size of 50, with beta scale values 0.1141548, 0.1812892, and 0.2423835

The example is based on Table II from
Glueck, D. H., & Muller, K. E. (2003). Adjusting power for a baseline covariate in linear models. Statistics in
Medicine, 22(16), 2535-2551."""

GAUSSIAN_TEST_8_TITLE="""GLMM(F, g) Example 8. Unconditional power for the uncorrected
univariate approach to repeated measures, Box, Geisser-Greenhouse,
and Huynh-Feldt tests, using Davies algorithm"""
GAUSSIAN_TEST_8_STUDY_DESIGN_DESCRIPTION="""The study design in Example 8 is a three sample design with a baseline covariate and four repeated measurements.
We calculate the unconditional power for a test of no di erence between groups at each time point. We calculate
unconditional power for the uncorrected univariate approach to repeated measures, Box, Geisser-Greenhouse,
and Huynh-Feldt tests.The exact distribution of the test statistic under the alternative hypothesis is obtained
using Davies' algorithm described in
Davies, R. B. (1980). Algorithm AS 155: The Distribution of a Linear Combination of Chi-Square Random
Variables. Applied Statistics, 29(3), 323-333.
Unconditional power is calculated for the following combinations of mean di erences and per group sample sizes.

1. Per group sample size of 5, with beta scale values 0.4997025, 0.8075886, and 1.097641
2. Per group sample size of 25, with beta scale values 0.1651525, 0.2623301, and 0.3508015
3. Per group sample size of 50, with beta scale values 0.1141548, 0.1812892, and 0.2423835

The example is based on Table II from
Glueck, D. H., & Muller, K. E. (2003). Adjusting power for a baseline covariate in linear models. Statistics in
Medicine, 22(16), 2535-2551."""

INTRODUCTION = """The following report contains validation results for the JavaStatistics library, a component of the GLIMMPSE
software system. For more information about GLIMMPSE and related publications, please visit\\n

https://samplesizeshop.org.\\n

The automated validation tests shown below compare power values produced by the GLIMMPSE V3 to
published results and also to simulation. Sources for published values include POWERLIB (Johnson et al. 2007)
and a SAS IML implementation of the methods described by Glueck and Muller (2003).
Validation results are listed in Section 3 of the report. Timing results show the calculation and simulation times
for the overall experiment and the mean times per power calculation. Summary statistics show the maximum
absolute deviation between the power value calculated by GLIMMPSE V3, the JavaStatistics library and the results obtained from
SAS or via simulation. The table in Section 3.3 shows the deviation values for each individual power comparison."""


all_names = [HOMEWORK_1_FILENAME,HOMEWORK_2_FILENAME,HOMEWORK_3_FILENAME,HOMEWORK_4_FILENAME,HOMEWORK_5_FILENAME,TEST_1_FILENAME, TEST_2_FILENAME, TEST_3_FILENAME, TEST_4_FILENAME, TEST_5_FILENAME, TEST_6_FILENAME, TEST_7_FILENAME]
all_titles = [HOMEWORK_1_TITLE, HOMEWORK_2_TITLE, HOMEWORK_3_TITLE, HOMEWORK_4_TITLE, HOMEWORK_5_TITLE, TEST_1_TITLE, TEST_2_TITLE, TEST_3_TITLE, TEST_4_TITLE, TEST_5_TITLE, TEST_6_TITLE, TEST_7_TITLE]
all_descriptions = [HOMEWORK_1_STUDY_DESIGN_DESCRIPTION, HOMEWORK_2_STUDY_DESIGN_DESCRIPTION, HOMEWORK_3_STUDY_DESIGN_DESCRIPTION, HOMEWORK_4_STUDY_DESIGN_DESCRIPTION, HOMEWORK_5_STUDY_DESIGN_DESCRIPTION, TEST_1_STUDY_DESIGN_DESCRIPTION, TEST_2_STUDY_DESIGN_DESCRIPTION, TEST_3_STUDY_DESIGN_DESCRIPTION, TEST_4_STUDY_DESIGN_DESCRIPTION, TEST_5_STUDY_DESIGN_DESCRIPTION, TEST_6_STUDY_DESIGN_DESCRIPTION, TEST_7_STUDY_DESIGN_DESCRIPTION]
all_list_inputs = [homework1_list_inputs, homework2_list_inputs, homework3_list_inputs, homework4_list_inputs, homework5_list_inputs, test1_list_inputs, test2_list_inputs, test3_list_inputs, test4_list_inputs, test5_list_inputs, test6_list_inputs, test7_list_inputs]
all_timings = [homework1_timings, homework2_timings, homework3_timings, homework4_timings, homework5_timings, test1_timings, test2_timings, test3_timings, test4_timings, test5_timings,  test6_timings, test7_timings]
all_deviations = [homework1_deviations, homework2_deviations, homework3_deviations, homework4_deviations, homework5_deviations, test1_deviations, test2_deviations, test3_deviations, test4_deviations, test5_deviations, test6_deviations, test7_deviations]
all_results = [homework1_results, homework2_results, homework3_results, homework4_results, homework5_results, test1_results, test2_results, test3_results, test4_results, test5_results, test6_results, test7_results]


all_gaussian_names = [GAUSSIAN_TEST_1_FILENAME, GAUSSIAN_TEST_4_FILENAME, GAUSSIAN_TEST_5_FILENAME, GAUSSIAN_TEST_8_FILENAME]
all_gaussian_titles = [GAUSSIAN_TEST_1_TITLE,GAUSSIAN_TEST_4_TITLE,GAUSSIAN_TEST_5_TITLE,GAUSSIAN_TEST_8_TITLE,]
all_gaussian_descriptions = [GAUSSIAN_TEST_1_STUDY_DESIGN_DESCRIPTION,GAUSSIAN_TEST_4_STUDY_DESIGN_DESCRIPTION,GAUSSIAN_TEST_5_STUDY_DESIGN_DESCRIPTION,GAUSSIAN_TEST_8_STUDY_DESIGN_DESCRIPTION,]
all_gaussian_list_inputs = [gaussian_test1_list_inputs,gaussian_test4_list_inputs,gaussian_test5_list_inputs,gaussian_test8_list_inputs,]
all_gaussian_timings = [gaussian_test1_timings,gaussian_test4_timings,gaussian_test5_timings,gaussian_test8_timings,]
all_gaussian_deviations = [gaussian_test1_deviations,gaussian_test4_deviations,gaussian_test5_deviations,gaussian_test8_deviations,]
all_gaussian_results = [gaussian_test1_results,gaussian_test4_results,gaussian_test5_results,gaussian_test8_results,]


for n in all_names:
    write_tex_file(False,
        n,
        all_titles[all_names.index(n)],
        INTRODUCTION,
        all_descriptions[all_names.index(n)],
        all_list_inputs[all_names.index(n)],
        all_timings[all_names.index(n)],
        all_deviations[all_names.index(n)],
        all_results[all_names.index(n)])
    write_pdf(n)

write_tex_results_file(V3_RESULTS_TABLE)
write_pdf('V3ResultsTable.tex')

for n in all_gaussian_names:
     write_tex_file(True,
                    n,
                    all_gaussian_titles[all_gaussian_names.index(n)],
                    INTRODUCTION,
                    all_gaussian_descriptions[all_gaussian_names.index(n)],
                    all_gaussian_list_inputs[all_gaussian_names.index(n)],
                    all_gaussian_timings[all_gaussian_names.index(n)],
                    all_gaussian_deviations[all_gaussian_names.index(n)],
                    all_gaussian_results[all_gaussian_names.index(n)])
     write_pdf(n)
