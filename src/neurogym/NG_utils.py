import numpy as np
import torch
import neurogym as ngym


def load_dataset(config_file, args):
    dataset = ngym.Dataset(config_file, batch_size = args.batch_size, seq_len = args.seq_len)
    dataset.seed(args.seed)
    return dataset

def add_task_indicator(dataset, indicator):
    inputs, labels = dataset()
    inputs = np.insert(inputs, len(inputs[0][0]), indicator, axis=2)
    return torch.from_numpy(inputs).type(torch.float), labels