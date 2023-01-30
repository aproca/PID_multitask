import os 
import pandas as pd

from src.PID import extract, k_order, full_order, batch
from src.PID import PID_utils

def get_performance_info(model_PID, activations):
    model_PID['avg_reward'] = activations['avg_reward']
    model_PID['accuracy'] = activations['accuracy']
    return model_PID

def run_model(model_PID, model_df, args):
    activations = extract.get_activations(model_df, args)
    model_PID = get_performance_info(model_PID, activations)
    if args.compute_PID:
        if args.k_order > 0:
            print('Computing k-order')
            model_PID = k_order.get_korder(model_PID, activations, args)
        if args.full_order:
            print('Computing full order')
            model_PID = full_order.get_fullorder(model_PID, activations, args)
    return model_PID

def run_folder_models(args):
    # Loads files, runs PID, and saves for all models in a folder (i.e., all seeds)
    PID_folder = args.PID_out_dir + args.base_folder
    data_files = os.listdir(args.activation_out_dir + args.base_folder)
    if '.DS_Store' in data_files:
        data_files.remove('.DS_Store')
    data_files = sorted(sorted(data_files), key=len)

    print('Running PID for {}'.format(args.base_folder))
    for file in data_files:
        print('File {}'.format(file))
        df = pd.read_csv(args.activation_out_dir + args.base_folder + file)
        PID_file = PID_folder + file[:-4]
        model_PID = PID_utils.check_dict_exists(PID_file + '.json')
        model_PID = run_model(model_PID, df, args)
        model_PID = PID_utils.save_dict(model_PID, PID_file, PID_folder)

def run_batch(args):
    # Averages all models in a folder
    PID_folder = args.PID_out_dir + args.base_folder
    PID_files = os.listdir(PID_folder)
    if '.DS_Store' in PID_files:
        PID_files.remove('.DS_Store')
    if 'batch.json' in PID_files:
        PID_files.remove('batch.json')
    PID_files = sorted(sorted(PID_files), key=len)
    print(PID_folder)
    print(PID_files)
    
    model_list = []
    [model_list.append(PID_utils.load_dict(PID_folder + file)) for file in PID_files]

    df = pd.DataFrame(model_list)
    batch_PID = batch.compute_batch_average(df)
    PID_utils.save_dict(batch_PID, PID_folder + 'batch', PID_folder)