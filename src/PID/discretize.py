import numpy as np

def get_bins(max_bin, num_bins):
    # generates bins for discretizing network activations
    return np.linspace(0, max_bin, num_bins)

def discretize_AAI_input(activations, args): 
    d_input = []
    ray_bins = get_bins(args.ray_max_bin, args.ray_num_bins)
    pos_bins = get_bins(args.pos_max_bin, args.pos_num_bins)
    num_rays = int((len(activations['inputs'][0]) - 7)/8)
    
    for i in range(len(activations['inputs'])):
        d_activation = []
        pos = np.array(activations['inputs'])[i,-3:]
        d_pos = np.digitize(pos, pos_bins, right=True).flatten()
        d_activation.append(''.join(str(k) for k in list(d_pos)))

        for j in range(num_rays):
            ray = np.array(activations['inputs'])[i, list(range(j*8, j*8 + 8))]
            onehot_ray = np.array(ray[0:-1]).astype(int)
            dist_ray = np.digitize(ray[7], ray_bins, right=True).flatten()
            d_ray = list(onehot_ray) + list(dist_ray)
            d_activation.append(''.join(str(k) for k in d_ray))
        d_input.append(np.array(d_activation).flatten())
    activations['inputs'] = d_input
    return activations

def discretize_input(activations): 
    d_input = []
    for act in activations['inputs']:
        d_activation = [str(j) for j in act]
        d_input.append(d_activation)
    activations['inputs'] = np.array(d_input)
    return activations

def discretize_FF_layers(activations, args): 
    # discretizes activations in each layer by binning based on value
    d_layers = []
    layer_bins = get_bins(args.layer_max_bin, args.layer_num_bins)
    for layer in activations['layers']:
        d_layer = []
        for i in range(len(layer)):
            d_activation = np.digitize(layer[i], layer_bins, right=True).flatten()
            d_layer.append([str(j) for j in d_activation])
        d_layers.append(np.array(d_layer))
    activations['layers'] = d_layers
    return activations

def discretize_activations(activations, args):
    if args.experiment == 'logic':
        activations = discretize_input(activations)
    elif args.experiment == 'animalai':
        activations = discretize_AAI_input(activations, args)
    
    activations = discretize_FF_layers(activations, args)

    return activations