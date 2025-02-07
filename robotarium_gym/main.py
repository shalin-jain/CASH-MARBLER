import yaml
import os
from robotarium_gym.utilities.misc import run_env, load_env_and_model, objectview
import argparse


def main():
    #The next three lines are checking if the program is being run in the normal setup or on the Robotarium (after using generate_submission.py)
    module_dir = os.path.dirname(__file__)
    if module_dir.split("/")[-1] != "robotarium_gym":
        module_dir = ""
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='PredatorCapturePrey', help='scenario name')
    args = parser.parse_args()

    if module_dir == "":
        config_path = "config.yaml"
    else:
        config_path = os.path.join(module_dir, "scenarios", args.scenario, "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = objectview(config)

    run_env(config, module_dir)
    # load_env_and_model(config, module_dir)

if __name__ == '__main__':
    main()