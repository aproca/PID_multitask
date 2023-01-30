import numpy as np
import torch

from src.logic import test_logic
from src.PID import PID_utils, k_order, runner

def compute_source_avg(pair_dict, idx, s_key):
    # computes average 2-order synergy for a particular source
    avg = 0
    for i in range(len(idx)):
        string = s_key + '_syn'
        for j in range(len(idx[i])):
            string = string + '_' + str(idx[i][j])
        syn = pair_dict[string]
        avg += syn
    avg = avg/len(idx)
    return avg

def get_source_avg_syn(args, s_key):
    # computes average 2-order synergy for every source in layer
    # only hidden layer sources
    model_PID = PID_utils.load_dict(args.PID_out_dir \
        + args.base_folder + args.base_file + '.json')
    pair_dict = model_PID['k2_x2h_tuples_syn']
    avg_pairwise = []
    
    # for each layer
    for i in range(len(args.layer_sizes)):
        source_avg = np.zeros(args.layer_sizes[i])
        korder_tuples = np.array(k_order.get_korder_tuples(args.layer_sizes[i], args))
        # for each source in each layer
        for j in range(args.layer_sizes[i]):
            source_pair = korder_tuples[np.where(korder_tuples == j)[0]]
            source_avg[j] = compute_source_avg(pair_dict[i+1], source_pair, s_key)
        avg_pairwise.append(source_avg)
    return avg_pairwise

def get_mask(source_avg, num_lesion, args):
    # creates masks to remove max/min synergy source
    max_mask = []
    min_mask = []
    for i in range(len(args.layer_sizes)):
        max_mask.append([1]*args.layer_sizes[i])
        min_mask.append([1]*args.layer_sizes[i])
        lesion_idx = np.argsort(source_avg[i])
        for j in range(num_lesion):
            max_mask[i][int(lesion_idx[-j-1])] = 0
            min_mask[i][int(lesion_idx[j])] = 0
    return max_mask, min_mask

def apply_lesion(args, source_avg, s_key, data):
    # iteratively lesions the max/min synergistic source in each layer
    orig_fold = args.base_folder[:-1]
    for num_lesion in range(1, args.layer_sizes[0] + 1):
        max_mask, min_mask = get_mask(source_avg, num_lesion, args)
        model = torch.load(args.model_out_dir + orig_fold + '/' + args.base_file)
        model.lesion = True
        model.mask = max_mask
        args.base_folder = '{}_max_mask_{}{}/'.format(orig_fold, s_key, num_lesion)
        test_logic.test(model, args, data)
        model.mask = min_mask
        args.base_folder = '{}_min_mask_{}{}/'.format(orig_fold, s_key, num_lesion)
        test_logic.test(model, args, data)


def lesion_experiment(args, dropout_p, seeds, data):
    syn_keys = ['nimin', 'nimmi'] 

    for s_key in syn_keys:
        for p in dropout_p:
            for s in seeds:
                args.base_folder = str(p) + '/'
                args.base_file = str(s)
                source_avg = get_source_avg_syn(args, s_key)
                apply_lesion(args, source_avg, s_key, data)
    
    for s_key in syn_keys:
        for p in dropout_p:
            args.base_folder = str(p) + '/'
            orig_fold = args.base_folder[:-1]
            for num_lesion in range(1, args.layer_sizes[0] + 1):
                args.base_folder = '{}_max_mask_{}{}/'.format(orig_fold, s_key, num_lesion)
                runner.run_folder_models(args)
                runner.run_batch(args)

                args.base_folder = '{}_min_mask_{}{}/'.format(orig_fold, s_key, num_lesion)
                runner.run_folder_models(args)
                runner.run_batch(args)
