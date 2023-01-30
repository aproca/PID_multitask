import numpy as np
import dit

def create_distribution(source, target):
    # creates distribution for discrete measures
    samples = []
    num_sources = len(source[0])
    for i in range(len(source)): # for each sample
        source_str = source[i][0]
        len_sources = [len(source_str)]
        for j in range(1, num_sources): # for each dimension of sample (string)
            source_str = source_str + source[i][j]
            len_sources.append(len(source[i][j]))
        source_str = [source_str]
        source_size = len(source_str[0])
        st_sample = ''.join([str(j) for j in list(np.concatenate((source_str, target[i])))])
        samples.append(st_sample)      
    values, counts = np.unique(samples, return_counts = True)
    pmf = []
    for i in range(len(counts)):
        pmf.append(counts[i]/np.sum(counts))
    distribution = []
    for i in range(len(values)):
        unq_samp = [values[i][0:len_sources[0]]]
        idx = len_sources[0]
        for j in range(1, num_sources): # splitting sources & targets
            unq_samp.append(values[i][idx:len_sources[j] + idx]) 
            idx = len_sources[j] + idx
        unq_samp.append(values[i][source_size:]) 
        distribution.append(tuple(unq_samp))      
    dit_distribution = dit.Distribution(distribution, pmf)    
    return dit_distribution

def remove_copies(x):
    is_unique = np.zeros_like(x, dtype=bool)
    is_unique[np.unique(x, return_index=True)[1]] = True
    return x[is_unique]

def create_data_matrix(source, target):
    # creates data matrix for continuous measures
    if not isinstance(source, np.ndarray):
        source = np.array(source)
    if not isinstance(target, np.ndarray):
        target = np.array(target)

    source_list = []
    for s in source.T:
        uniq_s = remove_copies(s)
        if len(uniq_s) > 1:
            source_list.append(s)
    if source_list == []:
        return None, None, None
    elif len(source_list) == 1:
        return None, None, None
    source_list = np.array(source_list)
    source = source_list.T

    target_list = []
    for t in target.T:
        uniq_t = remove_copies(t)
        if len(uniq_t) > 1:
            target_list.append(t)
    if target_list == []:
        return None, None, None 
    elif len(target_list) == 1:
        return None, None, None
    target_list = np.array(target_list)
    target = target_list.T
    
    X = np.concatenate((source, target), axis=1)
    num_sources = source.shape[1]
    src = []
    for i in range(0, num_sources):
        src.append([i])
    tgt = list(range(num_sources, num_sources + target.shape[1]))
    return X, src, tgt