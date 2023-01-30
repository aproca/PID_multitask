from scipy import stats 
import numpy as np

def compute_batch_average(df):
    # Computes mean/SEM of measures across all seeds
    batch_PID = {}
    for i in df:
        if type(df[i][0]) == list:
            if type(df[i][0][0]) == dict:
                pass
            else:
                try:
                    measure = np.mean(df[i].tolist(), axis = 0)
                except:
                    pass
                else:
                    if type(measure) == np.ndarray:
                        batch_PID['{}_avg'.format(i)] = list(measure)
                        batch_PID['{}_sem'.format(i)] = list(stats.sem(df[i].tolist(), axis = 0))
        else:
            measure = np.mean(df[i].tolist(), axis = 0)
            batch_PID['{}_avg'.format(i)] = measure
            batch_PID['{}_sem'.format(i)] = stats.sem(df[i].tolist(), axis=0)
    return batch_PID