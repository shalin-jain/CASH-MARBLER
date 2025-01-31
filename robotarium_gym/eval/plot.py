import wandb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def save_dataframe(df, filename):
    df.to_pickle(filename)
    print(f"DataFrame saved to {filename}")

def load_dataframe(filename):
    if os.path.exists(filename):
        df = pd.read_pickle(filename)
        print(f"DataFrame loaded from {filename}")
        df.index = range(0, len(df))
        print(df.head())
        return df
    else:
        print(f"File {filename} not found.")
        return None

def parse_wandb_data(data, metrics):
    metric_dict = {key: [] for key in metrics}
    for row in data:
        for key in metrics:
            metric_dict[key].append(row.get(key, None))
    return pd.DataFrame(metric_dict)

def fetch_wandb_data(project_name, tags, metrics):
    api = wandb.Api()
    runs = api.runs(project_name, {"tags": {"$in": tags}})
    print(f"found {len(runs)} runs with tags {tags}")
    
    all_run_data = []
    for run in tqdm(runs):
        hyperaware = True if run.config.get("agent") == "hyper_rnn" else False
        cap_aware = True if run.config.get("capability_aware") else False

        history = run.scan_history()
        run_data = parse_wandb_data(history, metrics)

        run_data['run_id'] = run.id
        run_data['run_name'] = run.name
        run_data['tags'] = ', '.join(run.tags)
        run_data['hyperaware'] = hyperaware
        run_data['cap_aware'] = cap_aware
        run_data['timestep'] = run_data['_step']
        run_data['seed'] = run.config.get("seed")

        all_run_data.append(run_data)
    
    return pd.concat(all_run_data, ignore_index=True)

def get_from_wandb():
    project_name = "CASH-MARBLER"
    tags = ['PCP-HARD-2']
    metrics = ['return_mean', 'return_std', 'test_return_mean', 'test_return_std', '_step']

    df = fetch_wandb_data(project_name, tags, metrics)
    filename = f"{tags[0]}.pkl" 
    save_dataframe(df, filename)

    print("saved")
    print(df.head())

def baseline_name(row):
    if row['hyperaware']: 
        if row['cap_aware']:
            return "CASH"
    else:
        if row['cap_aware']:
            return "RNN-EXP"
        else:
            return "RNN-IMP"

def combine_stats_across_seeds(df):
    grouped = df.groupby(['timestep', 'hyperaware', 'cap_aware'])
    
    def combine_stats(group):
        n_return = group['return_mean'].notna().sum()
        n_test = group['test_return_mean'].notna().sum()
        
        if n_return > 0:
            means = group['return_mean'].dropna()
            stds = group['return_std'].dropna()
            total_n = n_return * 32
            mean_return = means.mean()
            pooled_var = np.sum((32 - 1) * stds**2) / (total_n - n_return)
            std_return = np.sqrt(pooled_var)
        else:
            mean_return = np.nan
            std_return = np.nan
        
        if n_test > 0:
            test_means = group['test_return_mean'].dropna()
            test_stds = group['test_return_std'].dropna()
            total_n_test = n_test * 20
            mean_test_return = test_means.mean()
            pooled_var_test = np.sum((20 - 1) * test_stds**2) / (total_n_test - n_test)
            std_test_return = np.sqrt(pooled_var_test)
        else:
            mean_test_return = np.nan
            std_test_return = np.nan
        
        return pd.Series({
            'combined_return_mean': mean_return,
            'combined_return_std': std_return,
            'combined_test_return_mean': mean_test_return,
            'combined_test_return_std': std_test_return,
        })
    
    result = grouped.apply(combine_stats).reset_index()
    print(result.head())
    return result

def smooth_and_downsample(df, y_column, mean_window=50, std_window=50, downsample_factor=10):
    """
    Creates a new dataframe with smoothed and downsampled data, with separate
    smoothing controls for mean and standard deviation
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    y_column : str
        Column to analyze
    mean_window : int
        Window size for smoothing the mean
    std_window : int
        Window size for smoothing the standard deviation
    downsample_factor : int
        Factor by which to downsample the data
    """
    smoothed_data = []
    df_copy = df.copy()
    
    for baseline in df_copy['baseline'].unique():
        baseline_data = df_copy[df_copy['baseline'] == baseline].copy()
        baseline_data = baseline_data.sort_values('timestep')

        # grouped = baseline_data.groupby('timestep')[y_column] # .agg(['mean', 'std']).reset_index()
        # max_timesteps = baseline_data['timestep'].max()
        # max_timesteps_rows = baseline_data[baseline_data['timestep'] == max_timesteps]
        # cols = [y_column, 'baseline', 'run_name']
        # print("ALL SEEDS final success rates:")
        # print(max_timesteps_rows[cols])
        # print()
        
        # Group by timestep to calculate mean and std
        # grouped = baseline_data.groupby('timestep')[y_column].agg(['mean', 'std']).reset_index()
        
        # Smooth mean and std separately
        baseline_data['smooth_mean'] = baseline_data[f'{y_column}_mean'].rolling(
            window=mean_window, min_periods=1, center=True).mean()
        baseline_data['smooth_std'] = baseline_data[f'{y_column}_std'].rolling(
            window=std_window, min_periods=1, center=True).mean()
        
        # Downsample
        baseline_data = baseline_data.iloc[::downsample_factor]
        
        # Create dataframe with smoothed mean and smoothed std
        smoothed_df = pd.DataFrame({
            'timestep': baseline_data['timestep'],
            f'{y_column}': baseline_data['smooth_mean'],
            f'{y_column}_std': baseline_data['smooth_std'],
            'baseline': baseline
        })
        
        smoothed_data.append(smoothed_df)
    
    return pd.concat(smoothed_data)

def plot_metrics(df, y_label, y_column, title, mean_window, std_window, downsample_factor, save_folder):
    smoothed_df = smooth_and_downsample(df, y_column=y_column, 
                                      mean_window=mean_window,
                                      std_window=std_window,
                                      downsample_factor=downsample_factor)
    print(smoothed_df.head())
    plt.figure(figsize=(4, 3))
    
    base_palette = sns.color_palette()
    palette = {
        'CASH': base_palette[2],
        'RNN-EXP': base_palette[0],
        'RNN-IMP': base_palette[1],
    }
    
    for baseline in smoothed_df['baseline'].unique():
        baseline_data = smoothed_df[smoothed_df['baseline'] == baseline]
        color = palette[baseline]
        
        plt.plot(baseline_data['timestep'], baseline_data[f'{y_column}'], 
                color=color, label=baseline, linewidth=2)
        
        plt.fill_between(baseline_data['timestep'],
                        baseline_data[y_column] - baseline_data[f'{y_column}_std'],
                        baseline_data[y_column] + baseline_data[f'{y_column}_std'],
                        color=color, alpha=0.2)
    
    plt.xlabel('Timestep')
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout(pad=0.5)
    plt.legend(loc='best')



if __name__ == "__main__":
    # get_from_wandb()
    df = load_dataframe("MT-FINAL.pkl")
    combined_df = combine_stats_across_seeds(df)
    combined_df['baseline'] = combined_df.apply(baseline_name, axis=1)
    print(combined_df.head())

    plot_metrics(combined_df, y_label='Training Return', y_column='combined_return', title='QMIX / Material Transport', mean_window=50, std_window=50, downsample_factor=10, save_folder='plots')
    plt.savefig('plots/mt_return.png')
    plt.show()

    # plot_metrics(combined_df, y_label='Test Return', y_column='combined_test_return', title='Test Return', mean_window=50, std_window=50, downsample_factor=10, save_folder='plots')
    # plt.savefig('plots/pcp_hard_test_return.png')
    # plt.show()
