import scipy.optimize as sciop
import inspect
from sic4dvar_functions.sic4dvar_helper_functions import *
from lib.lib_variables import equations_dict

def bounds_preparation(_dA, _w, equations_dict, equation='ManningLW'):
    epsilon = 0.01
    equations_dict[equation]['bounds'] = {}
    if 'a0' in equations_dict[equation]['parameters']:
        dA_error = np.where(_dA < params.valid_min_dA)
        _dA[dA_error] = np.nan
        bnds_min = -np.nanmin(_dA) + epsilon
        bnds_max = max(np.nanmax(_w) * 100, 10 * abs(bnds_min))
        equations_dict[equation]['bounds']['a0'] = tuple((bnds_min, bnds_max))
    if 'n' in equations_dict[equation]['parameters']:
        bnds_min, bnds_max = (0.001, 1)
        equations_dict[equation]['bounds']['n'] = tuple((bnds_min, bnds_max))
    if 'cf' in equations_dict[equation]['parameters']:
        bnds_min, bnds_max = (0.001, 1)
        equations_dict[equation]['bounds']['cf'] = tuple((bnds_min, bnds_max))
    if 'alpha' in equations_dict[equation]['parameters']:
        bnds_min, bnds_max = (0.001, 1)
        equations_dict[equation]['bounds']['alpha'] = tuple((bnds_min, bnds_max))
    if 'beta' in equations_dict[equation]['parameters']:
        bnds_min, bnds_max = (-1, -0.0001)
        equations_dict[equation]['bounds']['beta'] = tuple((bnds_min, bnds_max))
    return equations_dict

def algo5(_q_ref, _dA, _w, _s, reach_id, equations_dict=equations_dict, equation='ManningLW', _z=[], param_dict=[]):
    equations_dict = bounds_preparation(_dA, _w, equations_dict, equation)
    data_to_use_dict = {}
    data_to_use_dict['q'] = np.copy(_q_ref)
    data_to_use_dict['abar'] = np.copy(_dA.data)
    data_to_use_dict['w'] = np.copy(_w.data)
    data_to_use_dict['s'] = np.copy(_s.data)
    if np.array(_z).size > 0:
        data_to_use_dict['wse'] = np.copy(_z.data)
    bnds = tuple((equations_dict[equation]['bounds'][name] for name in equations_dict[equation]['parameters']))
    if params.cython_version:
        results = sciop.differential_evolution(calc.objective_internal_data_cy, bounds=bnds, args=data_grouped_tuple, polish=True)
        a0, n = (results.x[0], results.x[1])
        q_est = np.copy(_q_ref)
        for i in range(q_est.shape[0]):
            if (data_to_use[i, 3] <= 0.0 or calc.check_na(data_to_use[i, 3])) or (data_to_use[i, 2] <= 0.0 or calc.check_na(data_to_use[i, 2])) or (data_to_use[i, 1] <= params.valid_min_dA or calc.check_na(data_to_use[i, 1])):
                q_est[i] = np.nan
            else:
                q_est[i] = cy.calc_q_manning_strickler_with_n(a0, data_to_use[i, 1], data_to_use[i, 3], n, data_to_use[i, 2])
    else:
        data_to_use_tuple = tuple(data_to_use_dict.values())
        args = (equation, data_to_use_tuple, tuple(equations_dict[equation]['bounds'].keys()))
        if False:
            results = sciop.differential_evolution(calc.objective_internal_data_Q_any, bounds=bnds, args=args, polish=True, strategy='best1bin')
        else:
            initial_guesses = []
            for key in list(equations_dict[equation]['bounds'].keys()):
                initial_guesses.append((equations_dict[equation]['bounds'][key][1] - equations_dict[equation]['bounds'][key][0]) / 2)
            print('BOUNDS:', bnds[0])
            new_lb = [bnds[0][0], bnds[1][0]]
            new_ub = [bnds[0][1], bnds[1][1]]
            results = sciop.least_squares(calc.objective_internal_data_Q_any, x0=initial_guesses, bounds=(new_lb, new_ub), args=args)
        results_dict = {}
        for j in range(0, len(results.x)):
            results_dict[list(equations_dict[equation]['bounds'].keys())[j]] = results.x[j]
        logging.info(f'Convergence success with strat best1bin: {results.success}')
        q_est = np.copy(_q_ref)
        for i in range(q_est.shape[0]):
            if (data_to_use_dict['s'][i] <= 0.0 or calc.check_na(data_to_use_dict['s'][i])) or (data_to_use_dict['w'][i] <= 0.0 or calc.check_na(data_to_use_dict['w'][i])) or (data_to_use_dict['abar'][i] <= params.valid_min_dA or calc.check_na(data_to_use_dict['abar'][i])):
                q_est[i] = np.nan
            else:
                print(f'Compute {equation} equation with parameters !')
                if calc.check_suitability_any(results_dict, data_to_use_dict, equation, i):
                    print('Suitable for computation.')
                    params_func = inspect.signature(getattr(calc, equation)).parameters
                    param_names = list(params_func.keys())
                    function_params = np.zeros(len(param_names))
                    for j, parameter in enumerate(param_names):
                        index_param = np.where(np.array(param_names) == parameter)[0]
                        if np.array(index_param).size > 0:
                            if parameter in data_to_use_dict:
                                function_params[j] = data_to_use_dict[parameter][i]
                            elif parameter in results_dict:
                                function_params[j] = results_dict[parameter]
                    func_params_dict = dict(zip(param_names, function_params))
                    q_est[i] = getattr(calc, equation)(**func_params_dict)
                else:
                    q_est[i] = np.nan
    q_est_masked = q_est
    for bound in equations_dict[equation]['bounds'].keys():
        print(equations_dict[equation]['bounds'][bound][1])
        if results_dict[bound] < equations_dict[equation]['bounds'][bound][0]:
            results_dict[bound] = equations_dict[equation]['bounds'][bound][0]
            logging.info(f'{bound} outside of bounds, set to lower bound: {results_dict[bound]}')
        elif results_dict[bound] > equations_dict[equation]['bounds'][bound][1]:
            results_dict[bound] = equations_dict[equation]['bounds'][bound][1]
            logging.info(f'{bound} outside of bounds, set to upper bound: {results_dict[bound]}')
    return (q_est_masked, results_dict)