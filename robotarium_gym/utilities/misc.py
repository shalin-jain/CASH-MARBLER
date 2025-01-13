import numpy as np
import random
import os
import importlib
import json
import torch
from rps.utilities.misc import *
import imageio
import pickle
from tqdm import tqdm

# imports needed for logging
import tensorflow as tf
import datetime

def is_close( agent_poses, agent_index, prey_loc, sensing_radius):
    agent_pose = agent_poses[:2, agent_index]
    prey_loc = prey_loc.reshape((1,2))[0]
    dist = np.linalg.norm(agent_pose - prey_loc)
    return dist <= sensing_radius, dist

def get_nearest_neighbors(poses, agent, num_neighbors):
    N = poses.shape[1]
    agents = np.arange(N)
    dists = [np.linalg.norm(poses[:2,x]-poses[:2,agent]) for x in agents]
    mins = np.argpartition(dists, num_neighbors+1)
    return np.delete(mins, np.argwhere(mins==agent))[:num_neighbors]


def get_random_vel():
    '''
        The first row is the linear velocity of each robot in meters/second (range +- .03-.2)
        The second row is the angular velocity of each robot in radians/second
    '''
    linear_vel = random.uniform(0.05,0.2) 
    #* (1 if random.uniform(-1,1)>0 else -1)
    angular_vel = 0
    # angular_vel = random.uniform(0,10)
    return np.array([linear_vel,angular_vel]).T

def convert_to_robotarium_poses(locations):
    poses = np.array(locations)
    N = poses.shape[0]
    return np.vstack((poses.T, np.zeros(N)))

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
        self.__json__ = json.dumps(d, indent=4)

def generate_initial_locations(num_locs, width, height, thresh, start_dist=.3, spawn_left = True):
    '''
    generates initial conditions for the robots and prey
    spawns all of them left of thresh if spawn_left is true, spawns them all right if spawn_left is false
    '''
    poses = generate_initial_conditions(num_locs, spacing=start_dist, width=width, height=height)
    if spawn_left:
        for i in range(len(poses[0])):
            poses[0][i] -= (width/2 - thresh)
            poses[2][i] = 0
    else:
        for i in range(len(poses[0])):
            poses[0][i] += (width/2 - thresh)
            poses[2][i] = 0
    return poses

def load_env_and_model(args, module_dir):
    ''' 
    Helper function to load a model from a specified scenario in args
    '''
    if module_dir == "": #All the weird checks with module_dir are for checking if running normally or for the Robotarium (after running generate_submission.py)
        model_config =args.model_config_file
    else:
        model_config = os.path.join(module_dir, "scenarios", args.scenario, "models", args.model_config_file)
    model_config = objectview(json.load(open(model_config)))
    model_config.n_actions = args.n_actions

    if module_dir == "":
        model_weights = torch.load( args.model_file, map_location=torch.device('cpu'))
    else:
        model_weights = torch.load( os.path.join(module_dir,  "scenarios", args.scenario, "models", args.model_file),\
                         map_location=torch.device('cpu'))
    input_dim = args.n_inputs

    if module_dir == "":
        actor = importlib.import_module(args.actor_file)
    else:
        actor = importlib.import_module(f'robotarium_gym.utilities.{args.actor_file}')
    actor = getattr(actor, args.actor_class)
    
    model_config.n_agents = args.n_agents
    model = actor(input_dim, model_config)
    print("num_params: ", sum(p.numel() for p in model.parameters()))
    model.load_state_dict(model_weights)

    if module_dir == "":
        env_module = importlib.import_module(args.env_file)
    else:
        env_module = importlib.import_module(f'robotarium_gym.scenarios.{args.scenario}.{args.env_file}')
    env_class = getattr(env_module, args.env_class)
    env = env_class(args)

    model_config.shared_reward = args.shared_reward

    if args.enable_logging:
        current_folder = os.getcwd()
        log_path = os.path.join(current_folder, 'logs')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        
        # Initialize logging folders if it is enabled
        log_path = os.path.join(log_path, args.scenario)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + args.model_config_file[:-5]
        log_path = os.path.join(log_path, current_time)
        model_config.log_path = log_path

    if args.save_gif:
        current_folder = os.getcwd()
        gif_path = os.path.join(current_folder, 'gifs')
        if not os.path.exists(gif_path):
            os.makedirs(gif_path)

        gif_path = os.path.join(gif_path, args.scenario)
        if not os.path.exists(gif_path):
            os.makedirs(gif_path)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + args.model_config_file[:-5]
        gif_path = os.path.join(gif_path, current_time)
        model_config.gif_path = gif_path

    return env, model, model_config


def run_env(config, module_dir):
    env, model, model_config = load_env_and_model(config, module_dir)
    obs = np.array(env.reset())
    n_agents = len(obs)

    totalReward = []
    totalSteps = []
    totalCollisions = []
    totalBoundary = []
    currCollisions = 0
    currBoundary = 0
    totalDists = np.zeros((config.episodes, n_agents))

    if config.save_gif:
        frames = []

    try:
        for i in tqdm(range(config.episodes)):
            episodeReward = 0
            episodeSteps = 0
            episodeDistTravelled = np.zeros((n_agents))
            hs = np.array([np.zeros((model_config.hidden_dim, )) for i in range(n_agents)])
            for j in range(config.max_episode_steps+1):      
                if model_config.obs_agent_id: #Appends the agent id if obs_agent_id is true. TODO: support obs_last_action too
                    obs = np.concatenate([obs,np.eye(n_agents)], axis=1)

                #Gets the q values and then the action from the q values
                if 'NS' in config.actor_class:
                    q_values, hs = model(torch.Tensor(obs), torch.Tensor(hs.T))
                else:
                    q_values, hs = model(torch.Tensor(obs), torch.Tensor(hs))
                    
                actions = np.argmax(q_values.detach().numpy(), axis=1)

                obs, reward, done, info = env.step(actions)
                
                episodeDistTravelled += info['dist_travelled']

                if info is not None and 'frames' in info.keys():
                    frames.extend(info['frames'])

                if model_config.shared_reward:
                    episodeReward += reward[0]
                else:
                    episodeReward += sum(reward)
                if done[0]:
                    episodeSteps = j+1
                    if "collision" in env.env.errors:
                        allCollisions = sum(env.env.errors["collision"])
                        if allCollisions > currCollisions:
                            episodeSteps = config.max_episode_steps
                    if "boundary" in env.env.errors:
                        allBoundary = sum(env.env.errors["boundary"])
                        if allBoundary > currBoundary:
                            episodeSteps = config.max_episode_steps
                    break
            if episodeSteps == 0:
                episodeSteps = config.max_episode_steps
            
            episodeCollision = 0
            episodeBoundary = 0
            if "collision" in env.env.errors:
                allCollisions = sum(env.env.errors["collision"])
                if allCollisions > currCollisions:
                    episodeCollision = allCollisions - currCollisions
                    currCollisions = allCollisions
                    # print(currCollisions)
            if "boundary" in env.env.errors:
                allBoundary = sum(env.env.errors["boundary"])
                if allBoundary > currBoundary:
                    episodeBoundary = allBoundary - currBoundary
                    currBoundary = allBoundary
                    # print(currBoundary)

            # print('Episode', i+1)
            # print('Episode reward:', episodeReward)
            # print('Episode steps:', episodeSteps)
            # print('Episode collisions:', episodeCollision)
            # print('Episode distance travelled:', episodeDistTravelled)

            totalReward.append(episodeReward)
            totalSteps.append(episodeSteps)
            totalCollisions.append(episodeCollision)
            totalBoundary.append(episodeBoundary)
            totalDists[i,:] = episodeDistTravelled

            if config.show_figure_frequency != -1 and config.save_gif:
                gif_path = os.path.join(module_dir, config.eval_dir, f"{config.actor_class}_{config.scenario}")
                if not os.path.exists(gif_path):
                    os.makedirs(gif_path)
                path_gif = os.path.join(gif_path, f"episode_{i}.gif")
                imageio.mimsave(path_gif, frames, duration = 100,loop=0)

            obs = np.array(env.reset())
    except Exception as error:
        print(error)
    finally:
        if config.save_eval_output:
            metrics_path = os.path.join(module_dir, config.eval_dir, f"{config.actor_class}_{config.capability_aware}_{config.scenario}")
            if not os.path.exists(metrics_path):
                os.makedirs(metrics_path)
            metrics_path = os.path.join(metrics_path, "metrics.pkl")
            with open(metrics_path, "wb") as f:
                pickle.dump({
                    "totalReward": totalReward,
                    "totalSteps": totalSteps,
                    "totalCollisions": totalCollisions,
                    "totalBoundary": totalBoundary,
                }, f)

        print(f'\n[Reward] Mean: {np.mean(totalReward)}, Standard Deviation: {np.std(totalReward)}')
        print(f'[Steps] Mean: {np.mean(totalSteps)}, Standard Deviation: {np.std(totalSteps)}')
        print(f'[Collisions] Mean: {np.mean(totalCollisions)}, Standard Deviation: {np.std(totalCollisions)}')
        print(f'[Boundary] Mean: {np.mean(totalBoundary)}, Standard Deviation: {np.std(totalBoundary)}')
        print(f'[Dist] Mean: {np.mean(totalDists, axis=0)}, Standard Deviation: {np.std(totalDists)}')