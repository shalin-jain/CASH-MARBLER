# CASH-MARBLER
We implement the CASH Policy Architecture for Multi-Robot Coordination Tasks in MARL. It is required to install the MARBLER submodules and EPyMARL submodules.

## Reproducing Results
### Training Results
Reproduce the trained models. All trained models evaluated in the paper are saved under `robotarium_gym/scenarios/<scenario_name>/models/<scenario_abbrv>/`
- Training seeds: `0, 1, 2`
    - Need to be set in both `robotarium_gym` config and `epymarl` config
- Configs:
    - Material Transport: `robotarium_gym/scenarios/MaterialTransport/config.yaml`
    - Predator Capture Prey: `robotarium_gym/scenarios/PredatorCapturePrey/config.yaml`
- *Example* Train CASH on MaterialTransport:
    - Set `agent: "hyper_rnn"` in `epymarl/src/config/default.yaml`
    - Set the below in `epymarl/src/config/envs/gymma.yaml`
        ```
        env_args:
            key: robotarium_gym:MaterialTransport-v0
        ...
        ```
### Evaluation Results
Reproduce the evaluation tables in the paper. The outcomes of all simulated evaluations are saved under `robotarium_gym/eval/`.
- *Example* Evaluate CASH on MaterialTransport with 500 episodes
    - In `robotarium_gym/scenarios/MaterialTransport/config.yaml/` set the following:
        ```
        model_config_file: qmix_hyper_rnn.json
        model_dir: MT/HyperRNN
        actor_file: hyper_rnn_agent
        actor_class: HyperRNNAgent
        episodes: 500
        ```
    - `python -m robotarium_gym.main --scenario MaterialTransport`
- Evaluating online capability changes
    - `Degrade`
        ```
        # Material Transport
        power_decay: True
        decay_rate: 0.99

        # Predator Capture Prey
        sensor_decay: True
        decay_rate: 0.999
        ```
    - `Failure`
        ```
        # Material Transport
        motor_failure: True

        # Predator Capture Prey
        sensor_failure: True
        ```

### Deployment Results
All robotarium deployment files can be found under `robotarium_eval/robotarium_submissions`. To run a specific deployment experiment, upload the contents of the relevant folder to the [Robotarium](https://www.robotarium.gatech.edu/dashboard) and set main.py as the Main file.

### Saved Results
To directly see results referenced in paper
- Training Metrics: `robotarium_gym/<scenario_abbrv>.pkl
    - Can be plotted with `robotarium_gym/plot.py`
- Evaluation Metrics: `robotarium_gym/<scenario_abbrv>/<agent_type>_<capability_aware>_<scenario_name>/metrics.pkl
- Trained Models: `robotarium_gym/scenarios/<scenario_name>/models/<scenario_abbrv>/`


## Submodule - MARBLER: Multi-Agent RL Benchmark and Learning Environment for the Robotarium
### Installation Instructions
1. Create new Conda Environment: `conda create -n MARBLER python=3.8 && conda activate MARBLER`. 
- Note that python 3.8 is only chosen to ensure compatitbility with EPyMARL.
2. Download and Install the [Robotarium Python Simulator](https://github.com/robotarium/robotarium_python_simulator)
- As of now, the most recent commit our code works with is 6bb184e. The code will run with the most recent push to the Robotarium but it will crash during training.
3. Install our environment by running `pip install -e .` in this directory
4. To test successfull installation, run `python3 -m robotarium_gym.main` to run a pretrained model

### Usage
* To look at current scenarios or create new ones or to evaluate trained models, look at the README in robotarium_gym
* To upload the agents to the Robotarium, look at the README in robotarium_eval

## Submodule - EPyMARL
1. Download and Install [EPyMARL](https://github.com/uoe-agents/epymarl). On Ubuntu 22.04, to successfully install EPyMARL, I have to: 
    - Checkout `v.1.0.0`
    - Downgrade `pip` to 24.0
    - Downgrade `wheel` to 0.38.4
    - Downgrade `setuptools` to 65.5.0
    - Install `einops` and `torchscatter`
2. Train agents normally using our gym keys
- For example: `python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=1000 env_args.key="robotarium_gym:PredatorCapturePrey-v0"`
- To train faster, ensure `robotarium` is False, `real_time` is False, and `show_figure_frequency` is large or -1 in the environment's `config.yaml`
- Known error: if `env_args.time_limit<max_episode_steps`, EPyMARL will crash after the first episode
3. Copy the trained weights to the models folder for the scenario that was trained
- Requires the agent.th file (location should be printed in the cout of the terminal the model was trained in, typically in EPyMARL/results/models/...)
- Requires the config.json file (typically in EPyMARL/results/algorithm_name/gym:scenario/...)
4. Update the scenario's config.yaml to use the newly trained agents


## Citing MARBLER and EPyMARL
* MARBLER:
>R. J. Torbati, S. Lohiya, S. Singh, M. S. Nigam and H. Ravichandar, "MARBLER: An Open Platform for Standardized Evaluation of Multi-Robot Reinforcement Learning Algorithms," 2023 International Symposium on Multi-Robot and Multi-Agent Systems (MRS), Boston, MA, USA, 2023, pp. 57-63, doi: 10.1109/MRS60187.2023.10416792.
* The Robotarium: 
>S. Wilson, P. Glotfelter, L. Wang, S. Mayya, G. Notomista, M. Mote, and M. Egerstedt. The robotarium: Globally impactful opportunities, challenges, and lessons learned in remote-access, distributed control of multirobot systems. IEEE Control Systems Magazine, 40(1):26–44, 2020.
* EPyMARL:
>Papoudakis, Georgios, et al. "Benchmarking multi-agent deep reinforcement learning algorithms in cooperative tasks." arXiv preprint arXiv:2006.07869 (2020).
