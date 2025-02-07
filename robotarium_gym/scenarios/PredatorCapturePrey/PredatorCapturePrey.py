import numpy as np
from gym import spaces
import copy
import yaml
import os

#This file should stay as is when copied to robotarium_eval but local imports must be changed to work with training!
from robotarium_gym.utilities.roboEnv import roboEnv
from robotarium_gym.utilities.misc import *
from robotarium_gym.scenarios.PredatorCapturePrey.visualize import *
from robotarium_gym.scenarios.base import BaseEnv
from robotarium_gym.scenarios.PredatorCapturePrey.agent import Agent

class PredatorCapturePrey(BaseEnv):
    def __init__(self, args):
        # Settings
        self.args = args

        self.num_robots = args.predator + args.capture
        self.agent_poses = None # robotarium convention poses
        self.prey_loc = None

        self.num_prey = args.num_prey
        self.num_predators = args.predator
        self.num_capture = args.capture

        if self.args.seed != -1:
             np.random.seed(self.args.seed)
        
        self.action_id2w = {0: 'left', 1: 'right', 2: 'up', 3:'down', 4:'no_action'}

        # if self.args.capability_aware:
        #     self.agent_obs_dim = 6
        # else:
        #     self.agent_obs_dim = 4
        self.agent_obs_dim = 6

        #Initializes the agents
        self.agents = []
        # Initialize predator agents
        for i in range(self.num_predators):
            predator_radius = np.random.choice(args.predator_radius)
            self.agents.append( Agent(i, predator_radius, 0, self.action_id2w, self.args.capability_aware) )
        # Initialize capture agents
        for i in range(self.num_capture):
            capture_radius = np.random.choice(args.capture_radius)
            self.agents.append( Agent(i + self.args.predator, 0, capture_radius, self.action_id2w, self.args.capability_aware) )

        #initializes the actions and observation spaces
        actions = []
        observations = []      
        for agent in self.agents:
            actions.append(spaces.Discrete(5))
            #Each agent's observation is a tuple of size self.agent_obs_dim
            obs_dim = self.agent_obs_dim * (self.args.num_neighbors + 1)
            #The lowest any observation will be is -5 (prey loc when can't see one), the highest is 3 (largest reasonable radius an agent will have)
            observations.append(spaces.Box(low=-5, high=3, shape=(obs_dim,), dtype=np.float32))        
        self.action_space = spaces.Tuple(tuple(actions))
        self.observation_space = spaces.Tuple(tuple(observations))

        self.visualizer = Visualize( self.args )
        self.env = roboEnv(self, args)

    def _generate_step_goal_positions(self, actions):
        '''
        User implemented
        Calculates the goal locations for the current agent poses and actions
        returns an array of the robotarium positions that it is trying to reach
        '''
        goal = copy.deepcopy(self.agent_poses)
        for i, agent in enumerate(self.agents):
            goal[:,i] = agent.generate_goal(goal[:,i], actions[i], self.args)
        return goal

    def _update_tracking_and_locations(self, agent_actions):
        # iterate over all the prey
        for i, prey_location in enumerate(self.prey_loc):
            # if the prey has already been captured, nothing to be done
            if self.prey_captured[i]:
                continue        
            #check if the prey has not been sensed
            if not self.prey_sensed[i]:
                # check if any of the agents has sensed it in the current step
                for agent in self.agents:
                    # check if any robot has it within its sensing radius
                    # print(self.agents.agent_poses[:2, agent.index], prey_location, np.linalg.norm(self.agents.agent_poses[:2, agent.index] - prey_location))
                    if np.linalg.norm(self.agent_poses[:2, agent.index] - prey_location) <= agent.sensing_radius:
                        self.prey_sensed[i] = True
                        break

            if self.prey_sensed[i]:
                # iterative over the agent_actions determined for each agent 
                for a, action in enumerate(agent_actions):
                    # check if any robot has no_action and has the prey within its capture radius if it is sensed already
                    if self.action_id2w[action]=='no_action'\
                        and np.linalg.norm(self.agent_poses[:2, self.agents[a].index] - prey_location) <= self.agents[a].capture_radius:
                        self.prey_captured[i] = True
                        break

    def _generate_state_space(self):
        '''
        Generates a dictionary describing the state space of the robotarium
        x: Poses of all the robots
        '''
        state_space = {}
        state_space['poses'] = self.agent_poses
        state_space['num_prey'] = self.num_prey - sum(self.prey_captured) # number of prey not captured
        state_space['unseen_prey'] = self.num_prey - sum(self.prey_sensed) # number of prey unseen till now 
        state_space['prey'] = []

        # return locations of all prey that are not captured  till now
        for i in range(self.num_prey):
            if not self.prey_captured[i]:
                state_space['prey'].append(np.array(self.prey_loc[i]).reshape((2,1)))
        return state_space
    
    def reset(self):
        '''
        Resets the simulation
        '''
        self.episode_steps = 0
        self.prey_locs = []
        self.num_prey = self.args.num_prey    

        # Resample agent capabilities
        self.agents = []
        # Initialize predator agents
        for i in range(self.num_predators):
            predator_radius = np.random.choice(self.args.predator_radius)
            self.agents.append( Agent(i, predator_radius, 0, self.action_id2w, self.args.capability_aware) )
        # Initialize capture agents
        for i in range(self.num_capture):
            capture_radius = np.random.choice(self.args.capture_radius)
            self.agents.append( Agent(i + self.args.predator, 0, capture_radius, self.action_id2w, self.args.capability_aware) )  
        
        # Agent locations
        width = self.args.ROBOT_INIT_RIGHT_THRESH - self.args.LEFT
        height = self.args.DOWN - self.args.UP
        self.agent_poses = generate_initial_locations(self.num_robots, width, height, self.args.ROBOT_INIT_RIGHT_THRESH, start_dist=self.args.start_dist)
        
        # Prey locations and tracking
        width = self.args.RIGHT - self.args.PREY_INIT_LEFT_THRESH
        self.prey_loc = generate_initial_locations(self.num_prey, width, height, self.args.ROBOT_INIT_RIGHT_THRESH, start_dist=self.args.step_dist, spawn_left=False)
        self.prey_loc = self.prey_loc[:2].T
        self.prey_captured = [False] * self.num_prey
        self.prey_sensed = [False] * self.num_prey
        
        self.state_space = self._generate_state_space()
        self.env.reset()
        return [[0]*(self.agent_obs_dim * (self.args.num_neighbors + 1))] * self.num_robots
        
    def step(self, actions_):
        '''
        Step into the environment
        Returns observation, reward, done, info (empty dictionary for now)
        '''
        terminated = False
        info = {}
        self.episode_steps += 1

        # call the environment step function and get the updated state
        return_message, dist, frames = self.env.step(actions_)
        
        self._update_tracking_and_locations(actions_)
        updated_state = self._generate_state_space()
        
        # get the observation and reward from the updated state
        obs     = self.get_observations(updated_state)
        if return_message != '':
            #print("Ending due to",return_message)
            info['message'] = return_message
            terminated =  True
            rewards = -5
        else:    
            rewards = self.get_rewards(updated_state)
            
            # condition for checking for the whether the episode is terminated
            if self.episode_steps > self.args.max_episode_steps or \
                updated_state['num_prey'] == 0:
                info['remaining'] = updated_state['num_prey']
                terminated = True              

        info['dist_travelled'] = dist
        if terminated:
            pass
            # print(f"Remaining prey: {updated_state['num_prey']} {return_message}")   
        
        if self.args.save_gif:
            info['frames'] = frames
        
        return obs, [rewards]*self.num_robots, [terminated]*self.num_robots, info

    def get_observations(self, state_space):
        '''
        Input: Takes in the current state space of the environment
        Outputs:
            an array with [agent_x_pos, agent_y_pos, sensed_prey_x_pose, sensed_prey_y_pose, sensing_radius, capture_radius]
            concatenated with the same array for the nearest neighbors based on args.delta or args.num_neighbors
        '''
        if self.prey_locs == []:
            for p in state_space['prey']:
                self.prey_locs = np.concatenate((self.prey_locs, p.reshape((1,2))[0]))
        # iterate over all agents and store the observations for each in a dictionary
        # dictionary uses agent index as key
        observations = {}
        capabilities = {}
        for agent in self.agents:
            full_obs = agent.get_observation(state_space, self.agents) 
            observations[agent.index] = full_obs[:-2]
            capabilities[agent.index] = full_obs[-2:]    
        
        full_observations = []
        for i, agent in enumerate(self.agents):
            full_observations.append(observations[agent.index])
            
            if self.args.num_neighbors >= self.num_robots-1:
                nbr_indices = [i for i in range(self.num_robots) if i != agent.index]
            else:
                nbr_indices = get_nearest_neighbors(state_space['poses'], agent.index, self.args.num_neighbors)

            # construct capability vector
            cap = capabilities[agent.index]
            nbr_cap = []
            for nbr in nbr_indices:
                nbr_cap.extend(capabilities[nbr])
            cap = np.concatenate((cap, nbr_cap))
            
            # full_observation[i] is of dimension [NUM_NBRS, OBS_DIM]
            for nbr_index in nbr_indices:
                full_observations[i] = np.concatenate( (full_observations[i],observations[nbr_index]) )
            full_observations[i] = np.concatenate( (full_observations[i], cap) )

        # dimension [NUM_AGENTS, NUM_NBRS, OBS_DIM]
        return full_observations

    def get_rewards(self, state_space):
        # Fully shared reward, this is a collaborative environment.
        reward = 0
        reward += (self.state_space['unseen_prey'] - state_space['unseen_prey']) * self.args.sense_reward
        reward += (self.state_space['num_prey'] - state_space['num_prey']) * self.args.capture_reward
        reward += self.args.time_penalty
        self.state_space = state_space
        return reward
    
    def get_action_space(self):
        return self.action_space
    
    def get_observation_space(self):
        return self.observation_space
