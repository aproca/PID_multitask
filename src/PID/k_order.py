import itertools
import numpy as np 
import random
from scipy import stats

from src.PID import compute_store, format_data

def get_korder_tuples(num_sources, args):
    # returns all k-order tuples
    indices = range(0, num_sources)
    korder_tuples = itertools.combinations(indices, args.k_order)
    korder_tuples = [list(i) for i in korder_tuples]
    return korder_tuples

def compute_korder(source, target, args, korder_tuples = None): # TODO: make sure '_' makes sense
    if args.discrete_PID:
        source = np.array(source)
        if korder_tuples is None:
            korder_tuples = get_korder_tuples(len(source[0]), args)
    else:
        X, src_idx, tgt_idx = format_data.create_data_matrix(source, target)
        if src_idx is None or tgt_idx is None or X is None:
            return {}
        korder_tuples = get_korder_tuples(len(src_idx), args)

    if len(korder_tuples) > 45 and args.efficient_korder:
        korder_tuples = random.sample(korder_tuples, 45)
    
    measure_list = dict()
    tuples = dict()
    for tup in korder_tuples:
        if args.discrete_PID:
            source_tup = source[:, tup]
            measures = compute_store.compute_discrete_measures(source_tup, target, args)
        else:
            source_tup = []
            [source_tup.append(src_idx[tup[i]]) for i in range(len(tup))] ## TODO: MAKE SURE THIS WORKS
            measures = compute_store.compute_continuous_measures(X, source_tup, tgt_idx, args)

        if len(measure_list) > 0 and len(measures) > 0:
            [measure_list[key].append(value) for key, value in measures.items()] # TODO: CHECK IF THIS IS CORRECT SYNTAX
        else:
            for key, value in measures.items():
                measure_list[key] = [value]

        if args.save_korder_tuples:
            tuple_str = '_' + '_'.join(str(i) for i in tup)
            for key, value in measures.items():
                if 'red' not in key:
                    tuples[key + tuple_str] = value
    
    avg_measures = dict()
    for key, value in measure_list.items():
        avg_measures[key + '_avg'] = np.mean(np.array(value))
        avg_measures[key + '_sem'] = stats.sem(np.array(value))
    if args.save_korder_tuples:
        avg_measures['tuples_syn'] = tuples
    return avg_measures

def compute_AAI_input_measures(source, target, args):
    measures = dict()
    # pairwise raycasts
    raycast_measures = compute_korder(np.array(source)[:,1:], target, args)
    for key, value in raycast_measures.items():
        measures['ray_' + key] = value

    # pairwise raycast/position
    korder_tuples = []
    num_raycasts = np.array(source).shape[1] - 1
    for i in range(1, num_raycasts + 1):
        korder_tuples.append((0, i))
    position_measures = compute_korder(source, target, args, korder_tuples)
    for key, value in position_measures.items():
        measures['pos_' + key] = value
    return measures

def get_korder_FF(model_PID, activations, args): # discrete only
    k_prefix = 'k{}_'.format(args.k_order)
    if args.experiment == 'animalai':
        measures = compute_AAI_input_measures(activations['inputs'], activations['layers'][0], args)
    else:
        measures = compute_korder(activations['inputs'], activations['layers'][0], args)
    model_PID = compute_store.store_measures(model_PID, args, k_prefix + 'x2h_', measures, True)

    for i in range(len(activations['layers'])):
        if i == len(activations['layers']) - 1:
            measures = compute_korder(activations['layers'][i], activations['outputs'], args)
        else:
            measures = compute_korder(activations['layers'][i], activations['layers'][i+1], args)
        model_PID = compute_store.store_measures(model_PID, args, k_prefix + 'x2h_', measures, False)
    
    return model_PID

def get_korder_RNN(model_PID, activations, args): # continuous only
    k_prefix = 'k{}_'.format(args.k_order)
    xh = np.concatenate((np.array(activations['inputs']), np.array(activations['hidden_t'])), axis = 1)
    measures = compute_korder(xh, activations['hidden_tp1'], args)
    model_PID = compute_store.store_measures(model_PID, args, k_prefix + 'xh2h_', measures, True)
    return model_PID

def get_korder(model_PID, activations, args):
    if args.network_type == 'feedforward' and args.discrete_PID:
        model_PID = get_korder_FF(model_PID, activations, args)
    elif args.network_type == 'recurrent' and not args.discrete_PID:
        model_PID = get_korder_RNN(model_PID, activations, args)
    return model_PID