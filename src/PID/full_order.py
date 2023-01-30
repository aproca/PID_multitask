import numpy as np

from src.PID import compute_store, format_data

def get_fullorder_FF(model_PID, activations, args):
    measures = compute_store.compute_discrete_measures(activations['inputs'], activations['layers'][0], args)
    model_PID = compute_store.store_measures(model_PID, args, 'x2h_', measures, True)

    for i in range(len(activations['layers'])):
        if i == len(activations['layers']) - 1:
            measures = compute_store.compute_discrete_measures(activations['layers'][i], activations['outputs'], args)
        else:
            measures = compute_store.compute_discrete_measures(activations['layers'][i], activations['layers'][i+1], args)
        model_PID = compute_store.store_measures(model_PID, args, 'x2h_', measures, False)
    return model_PID

def get_fullorder_RNN(model_PID, activations, args):
    # continuous only
    xh = np.concatenate((np.array(activations['inputs']), np.array(activations['hidden_t'])), axis = 1)
    X, src_idx, tgt_idx = format_data.create_data_matrix(xh, activations['hidden_tp1'])
    if src_idx is None or tgt_idx is None or X is None:
        return model_PID
    measures = compute_store.compute_continuous_measures(X, src_idx, tgt_idx, args)
    model_PID = compute_store.store_measures(model_PID, args, 'xh2h_', measures, True)
    return model_PID

def get_fullorder(model_PID, activations, args):
    if args.network_type == 'feedforward' and args.discrete_PID:
        model_PID = get_fullorder_FF(model_PID, activations, args)
    elif args.network_type == 'recurrent' and not args.discrete_PID:
        model_PID = get_fullorder_RNN(model_PID, activations, args)
    return model_PID