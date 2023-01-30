import torch 
from collections import namedtuple
import pandas as pd
import os
import numpy as np

def test(model, args, test_data):
    # tests model and saves activations
    transition = namedtuple('transition', ('observation', 'activation', 'reward'))
    configuration = namedtuple('configuration', ('episode', 'buffer', 'success'))
    model_info = namedtuple('model_info', ('configuration_file', 'configuration_buffer'))

    print('Testing on {}'.format(args.dataset))
    transition_buffer, configuration_buffer, model_buffer = [], [], []
    correct, total = 0, 0
    inputs, targets = [], []
    for sample in test_data:
        inputs.append(sample[0])
        targets.append(sample[1])

    model.eval()
    with torch.no_grad():
        for input, target in zip(inputs, targets):
            logit = model(input)
            predicted = np.argmax(logit.detach().numpy(), axis = -1)
            total += 1
            correct += len(np.where(predicted == target)[0])
            activation = model.activations
            reward = len(np.where(predicted == target)[0])
            transition_buffer.append(transition(input, activation, reward))
    print(f"Accuracy {correct/total}")  
    configuration_buffer.append(configuration(1, transition_buffer, correct))
    model_buffer.append(model_info('logic', configuration_buffer)) 
    df = pd.DataFrame.from_records(model_buffer)
    if not os.path.exists(args.activation_out_dir + args.base_folder):
        os.makedirs(args.activation_out_dir + args.base_folder)
    df.to_csv(args.activation_out_dir + args.base_folder + args.base_file + '.csv')