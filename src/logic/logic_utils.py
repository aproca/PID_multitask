import torch 
import itertools
import numpy as np
import os

def generate_binary_permutations(n):
    # creates binary permutations of length n for the datasets
    perm = itertools.product(range(2), repeat = n)
    perm = np.array([list(i) for i in perm])
    return perm

def generate_logic_data(args):
    inputs = generate_binary_permutations(args.input_size)
    labels = []
    if args.dataset == 'COPY':
        [labels.append(inputs[i][0]) for i in range(len(inputs))]
    elif args.dataset == 'XOR':
        for i in inputs:
            count = np.count_nonzero(i == 1)
            [labels.append(0) if count % 2 == 0 else labels.append(1)] 
    return inputs, labels

def dataloader_set(inputs, labels):
    # modifies dataset for dataloader
    data = []
    for input, label in zip(inputs, labels):
        data.append([input, label])
    return data

def get_data(args):
    inputs, labels = generate_logic_data(args)
    return dataloader_set(inputs, labels)
    
def save_model(model, folder, epoch):
    PATH = 'trained_models/{}model_{}'.format(folder, epoch)
    if not os.path.exists('trained_models/' + folder[:-1]):
        os.makedirs('trained_models/' + folder[:-1])
    torch.save(model, PATH) 

def accuracy(logits, targets):
    predicted = np.argmax(logits.detach().numpy(), axis = -1)
    return len(np.where(predicted == targets.detach().numpy())[0])/targets.size(0)