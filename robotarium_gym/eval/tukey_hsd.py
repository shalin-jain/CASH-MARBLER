import pandas as pd
import pickle
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def tukey_hsd_test(x, y, z, metric_name):
    data_x, method_x = x
    data_y, method_y = y
    data_z, method_z = z   
    combined_data = pd.concat([
        pd.DataFrame({metric_name: data_x, 'Method': method_x}),
        pd.DataFrame({metric_name: data_y, 'Method': method_y}),
        pd.DataFrame({metric_name: data_z, 'Method': method_z})
    ])
    
    # Print mean and standard deviation for each method
    print(f"\nMean and Std for {metric_name}:")
    for data, method in [x, y, z]:
        print(f"{method}: Mean = {np.mean(data):.4f}, Std = {np.std(data):.4f}")
    
    # Perform Tukey HSD test
    tukey_result = pairwise_tukeyhsd(endog=combined_data[metric_name], groups=combined_data['Method'], alpha=0.05)
    print(f"\nTukey HSD results for {metric_name}:")
    print(tukey_result)
    print("\nPairwise p-values:")
    print(tukey_result.pvalues)

if __name__ == "__main__":
    parent_folder = "PCP_SD/"
    scenario = "PredatorCapturePrey"
    # load data per method
    hyper_rnn = load_pkl(f"{parent_folder}HyperRNNAgent_True_{scenario}/metrics.pkl")
    # rnn_exp = load_pkl("HyperRNNAgent_MaterialTransport/metrics.pkl")
    # rnn_imp = load_pkl("HyperRNNAgent_MaterialTransport/metrics.pkl")
    rnn_exp = load_pkl(f"{parent_folder}RNNAgent_True_{scenario}/metrics.pkl")
    rnn_imp = load_pkl(f"{parent_folder}RNNAgent_False_{scenario}/metrics.pkl")
    
    # Extract metrics
    metrics = ['totalReward', 'totalSteps', 'totalCollisions', 'totalBoundary']
    
    for metric in metrics:
        tukey_hsd_test(
            (hyper_rnn[metric], "HyperRNN"),
            (rnn_exp[metric], "RNNEXP"),
            (rnn_imp[metric], "RNNIMP"),
            metric
        )