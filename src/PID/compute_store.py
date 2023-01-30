from src.PID import continuous_functions, format_data, discrete_functions
from src.PID import PID_utils

def compute_continuous_measures(X, src_idx, tgt_idx, args):
    if src_idx is None or tgt_idx is None or X is None:
        return {}
    measures = dict()
    measures['immi_syn'], measures['mi'] = continuous_functions.gc_immi_syn(X, src_idx, tgt_idx, True)
    measures['immi_syn'] = PID_utils.is_real_number(measures['immi_syn'])
    measures['mi'] = PID_utils.is_real_number(measures['mi'])
    measures['nimmi_syn'] = PID_utils.is_real_number(measures['immi_syn']/measures['mi'])

    if args.compute_redundancy:
        measures['immi_red'] = continuous_functions.gc_immi_syn(X, src_idx, tgt_idx, False)
        measures['immi_red'] = PID_utils.is_real_number(measures['immi_red'])
        measures['nimmi_red'] = PID_utils.is_real_number(measures['immi_red']/measures['mi'])
    return measures

def compute_discrete_measures(source, target, args):
    distribution = format_data.create_distribution(source, target)
    measures = dict()
    measures['imin_syn'], measures['mi'] = discrete_functions.imin_syn(distribution, True)
    measures['imin_syn'] = measures['imin_syn']
    measures['mi'] = measures['mi']
    measures['immi_syn'] = discrete_functions.immi_syn(distribution)
    measures['nimin_syn'] = PID_utils.is_real_number(measures['imin_syn']/measures['mi'])
    measures['nimmi_syn'] = PID_utils.is_real_number(measures['immi_syn']/measures['mi'])

    if args.compute_redundancy:
        measures['imin_red'] = discrete_functions.imin_red(distribution)
        measures['immi_red'] = discrete_functions.immi_red(distribution)     
        measures['nimin_red'] = PID_utils.is_real_number(measures['imin_red']/measures['mi'])
        measures['nimmi_red'] = PID_utils.is_real_number(measures['immi_red']/measures['mi'])
    return measures

def store_measures(model_PID, args, prefix, measures, initialize):
    for key, value in measures.items():
        if initialize:
            model_PID[prefix + key] = [value]
        else:
            if args.experiment == 'animalai':
                model_PID[prefix + 'ray_' + key].append(value)
                model_PID[prefix + 'pos_' + key].append(value)
            else:
                model_PID[prefix + key].append(value)
    return model_PID