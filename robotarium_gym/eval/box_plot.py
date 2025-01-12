import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def prepare_dataframe(data, method_name):
    df = pd.DataFrame(data)
    df['method'] = method_name
    return df

hyper_rnn = load_pkl("HyperRNNAgent_True_MaterialTransport/metrics.pkl")
# rnn_exp = load_pkl("HyperRNNAgent_MaterialTransport/metrics.pkl")
# rnn_imp = load_pkl("HyperRNNAgent_MaterialTransport/metrics.pkl")
rnn_exp = load_pkl("RNNAgent_True_MaterialTransport/metrics.pkl")
rnn_imp = load_pkl("RNNAgent_False_MaterialTransport/metrics.pkl")

# Convert each method's data into a DataFrame
df_hyper_rnn = prepare_dataframe(hyper_rnn, 'CASH')
df_rnn_exp = prepare_dataframe(rnn_exp, 'RNN-EXP')
df_rnn_imp = prepare_dataframe(rnn_imp, 'RNN-IMP')

base_palette = sns.color_palette()
palette = {
    'CASH': base_palette[2],
    'RNN-EXP': base_palette[0],
    'RNN-IMP': base_palette[1],
}

# Combine all data into one DataFrame
combined_df = pd.concat([df_hyper_rnn, df_rnn_exp, df_rnn_imp])

# Plotting each metric
for metric in ['totalReward', 'totalSteps', 'totalCollisions']:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='method', y=metric, data=combined_df, legend='auto', palette=palette)
    plt.title(f'Box Plot of {metric}')
    plt.ylabel(metric)
    plt.xlabel('Method')
    plt.grid(True)
    plt.savefig(f'box_plots/{metric}.png')
    plt.show()
