import os
import re
import numpy as np

def extract_metrics_from_file(filepath):
    metrics_pattern = {
        'reward': r"Episode reward: ([\d\.\-eE]+)",
        'steps': r"Episode steps: (\d+)",
        'collisions': r"Episode collisions: (\d+)",
        'boundary': r"Episode boundary: (\d+)"
    }
    metrics = {key: [] for key in metrics_pattern}

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
        for key, pattern in metrics_pattern.items():
            matches = re.findall(pattern, content)
            if key == 'reward':
                metrics[key].extend([float(m) for m in matches])
            else:
                metrics[key].extend([int(m) for m in matches])
    return metrics

def aggregate_metrics(log_folder):
    results = {}
    for method in os.listdir(log_folder):
        method_path = os.path.join(log_folder, method)
        if os.path.isdir(method_path):
            all_metrics = {key: [] for key in ['reward', 'steps', 'collisions', 'boundary']}
            for root, _, files in os.walk(method_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_metrics = extract_metrics_from_file(file_path)
                    for key in all_metrics:
                        all_metrics[key].extend(file_metrics[key])
            
            stats = {}
            for key, values in all_metrics.items():
                if values:
                    stats[key] = {
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }
                else:
                    stats[key] = {'mean': None, 'std': None}
            results[method] = stats
    return results

def print_results(results):
    for method, stats in results.items():
        print(f"Method: {method}")
        for metric, values in stats.items():
            mean = values['mean']
            std = values['std']
            if mean is not None:
                print(f"  {metric.capitalize()}: Mean = {mean:.3f}, Std = {std:.3f}")
            else:
                print(f"  {metric.capitalize()}: No data")
        print()

if __name__ == "__main__":
    log_folder = "robotarium-logs"  # Replace with your actual log folder path
    results = aggregate_metrics(log_folder)
    print_results(results)