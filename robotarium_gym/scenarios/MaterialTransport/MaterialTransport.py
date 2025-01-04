from gym import spaces
import numpy as np
import copy
from robotarium_gym.scenarios.base import BaseEnv
from robotarium_gym.utilities.misc import *
from robotarium_gym.scenarios.MaterialTransport.visualize import Visualize
from robotarium_gym.utilities.roboEnv import roboEnv


class Agent:
    #These agents are specifically implimented for the warehouse scenario
    def __init__(self, index, action_id_to_word, torque, speed):
        self.index = index
        self.torque = torque
        self.speed = speed
        self.load = 0
        self.action_id2w = action_id_to_word
   
    def generate_goal(self, goal_pose, action, args):    
        '''
        updates the goal_pose based on the agent's actions
        '''   
        action = action // 4 #This is to account for the messages
        if self.action_id2w[action] == 'left':
                goal_pose[0] = max( goal_pose[0] - self.speed, args.LEFT)
                goal_pose[1] = args.UP if goal_pose[1] < args.UP else \
                      args.DOWN if goal_pose[1] > args.DOWN else goal_pose[1]
        elif self.action_id2w[action] == 'right':
                goal_pose[0] = min( goal_pose[0] + self.speed, args.RIGHT)
                goal_pose[1] = args.UP if goal_pose[1] < args.UP else \
                      args.DOWN if goal_pose[1] > args.DOWN else goal_pose[1]
        elif self.action_id2w[action] == 'up':
                goal_pose[0] = args.LEFT if goal_pose[0] < args.LEFT else \
                      args.RIGHT if goal_pose[0] > args.RIGHT else goal_pose[0]
                goal_pose[1] = max( goal_pose[1] - self.speed, args.UP)
        elif self.action_id2w[action] == 'down':
                goal_pose[0] = args.LEFT if goal_pose[0] < args.LEFT else \
                      args.RIGHT if goal_pose[0] > args.RIGHT else goal_pose[0]
                goal_pose[1] = min( goal_pose[1] + self.speed, args.DOWN)
        else:
             goal_pose[0] = args.LEFT if goal_pose[0] < args.LEFT else \
                      args.RIGHT if goal_pose[0] > args.RIGHT else goal_pose[0]
             goal_pose[1] = args.UP if goal_pose[1] < args.UP else \
                      args.DOWN if goal_pose[1] > args.DOWN else goal_pose[1]
        
        return goal_pose

class MaterialTransport(BaseEnv):
    def __init__(self, args):
        self.args = args
        self.num_robots = self.args.n_agents
        self.agent_poses = None

        # Calculate the updated observation dimension
        # ego_pos_x, ego_pos_y, other_agents_pos_x * (n_agents - 1), other_agents_pos_y * (n_agents - 1)
        # zone1_load, zone2_load, ego_torque, ego_speed, other_agents_torque * (n_agents - 1), other_agents_speed * (n_agents - 1)
        self.agent_obs_dim = 2 + 2 * (self.num_robots - 1) + 2 + 2 + 2 * (self.num_robots - 1)

        self.zone1_args = copy.deepcopy(self.args.zone1)
        del self.zone1_args['distribution']   
        self.zone2_args = copy.deepcopy(self.args.zone2)
        del self.zone2_args['distribution']  

        if self.args.seed != -1:
             np.random.seed(self.args.seed)
        
        #This isn't really needed but makes a bunch of stuff clearer
        self.action_id2w = {0: 'left', 1: 'right', 2: 'up', 3:'down', 4:'no_action'}
        
        self.agents = []
        for i in range(self.args.n_fast_agents):
            self.agents.append(Agent(i, self.action_id2w, self.args.small_torque, self.args.fast_step))
        for i in range(self.args.n_fast_agents, self.args.n_fast_agents+self.args.n_slow_agents):
            self.agents.append(Agent(i, self.action_id2w, self.args.large_torque, self.args.slow_step))

        #Initializes the action and observation spaces
        actions = []
        observations = []
        for a in self.agents:
            actions.append(spaces.Discrete(20))
            #each agent's observation is a tuple of size 3
            #the minimum observation is the left corner of the robotarium, the maximum is the righ corner
            observations.append(spaces.Box(low=-1.5, high=1.5, shape=(self.agent_obs_dim,), dtype=np.float32))
        self.action_space = spaces.Tuple(tuple(actions))
        self.observation_space = spaces.Tuple(tuple(observations))
        
        self.visualizer = Visualize(self.args) #needed for Robotarium renderings
        #Env impliments the robotarium backend
        #It expects to have access to agent_poses, visualizer, num_robots and _generate_step_goal_positions
        self.env = roboEnv(self, args)  

    def reset(self):
        self.episode_steps = 0
        self.messages = [0,0,0,0] 

        #Randomly sets the load for each zone
        self.zone1_load = int(getattr(np.random, self.args.zone1['distribution'])(**self.zone1_args))
        self.zone2_load = int(getattr(np.random, self.args.zone2['distribution'])(**self.zone2_args))
        
        for a in self.agents:
            a.load=0
        
        #Generate the agent locations based on the config
        width = self.args.end_goal_width
        height = self.args.DOWN - self.args.UP
        #Agents can spawn in the Robotarium between UP, DOWN, LEFT and LEFT+end_goal_width for this scenario
        self.agent_poses = generate_initial_locations(self.num_robots, width, height, self.args.LEFT+self.args.end_goal_width, start_dist=self.args.start_dist)
        self.env.reset()
        return [[0]*self.agent_obs_dim] * self.num_robots
    
    def step(self, actions_):
        self.episode_steps += 1
        info = {}

        #Robotarium actions and updating agent_poses all happen here
        return_message, dist, frames = self.env.step(actions_)
        for i in range(len(self.messages)):
            self.messages[i] = actions_[i] % 4

        obs = self.get_observations()
        if return_message == '':
            reward = self.get_reward()       
            terminated = self.episode_steps > self.args.max_episode_steps #For this environment, episode only ends after timing out
            #Terminates when all agent loads are 0 and the goal zone loads are 0
            if not terminated:
                terminated = self.zone1_load == 0 and self.zone2_load == 0
                if terminated:
                    for a in self.agents:
                        if a.load != 0:
                            terminated = False
                            break
        else:
            #print("Ending due to", return_message)
            info['message'] = return_message
            reward = -6
            terminated = True
        
        info['dist_travelled'] = dist
        if terminated:
            print(f'Remaining: {self.zone1_load + self.zone2_load + sum(a.load for a in self.agents)} {return_message}')
            info['remaining'] = self.zone1_load + self.zone2_load + sum(a.load for a in self.agents)        
 
        if self.args.save_gif:
            info['frames'] = frames

        return obs, [reward] * self.num_robots, [terminated]*self.num_robots, info
    
    def get_observations(self):
        """
        Constructs observations for all agents.

        [ego_pos_x, ego_pos_y, *other_agents_pos_x, *other_agents_pos_y, zone1_load, zone2_load, 
         ego_torque, ego_speed, *other_agents_torque, *other_agents_speed]
        
        if capability unaware, speeds and torques will be set to 0 trivially
        """
        observations = []
        for ego_index, ego_agent in enumerate(self.agents):
            ego_pos = self.agent_poses[:, ego_index][:2]  # Ego position (x, y)
            other_agents_pos = [
                self.agent_poses[:, i][:2] for i in range(self.num_robots) if i != ego_index
            ]  # other agents' positions
            other_agents_pos_flat = [coord for pos in other_agents_pos for coord in pos]

            if self.args.capability_aware: 
                other_agents_torque = [
                    self.agents[i].torque for i in range(self.num_robots) if i != ego_index
                ]  # other agents' torque

                other_agents_speed = [
                    self.agents[i].speed for i in range(self.num_robots) if i != ego_index
                ]  # other agents' speed
            else:
                other_agents_torque = [0] * (self.num_robots - 1)
                other_agents_speed = [0] * (self.num_robots - 1)

            observation = [
                *ego_pos,
                *other_agents_pos_flat,
                self.zone1_load,
                self.zone2_load,
                ego_agent.torque if self.args.capability_aware else 0,
                ego_agent.speed if self.args.capability_aware else 0,
                *other_agents_torque,
                *other_agents_speed,
            ]
            observations.append(observation)

        return observations

    def get_reward(self):
        '''
        Agents take a small penalty every step and get a reward when they unload proportional to how much load they are carrying
        '''
        reward = self.args.time_penalty
        for a in self.agents:
            pos = self.agent_poses[:, a.index ][:2]
            if a.load > 0:             
                if pos[0] < -1.5 + self.args.end_goal_width:
                    reward += a.load * self.args.unload_multiplier
                    a.load = 0
            else:
                if pos[0] > 1.5 - self.args.end_goal_width:
                    if self.zone2_load > a.torque:              
                        a.load = a.torque
                        self.zone2_load -= a.torque
                    else:
                        a.load = self.zone2_load
                        self.zone2_load = 0
                    reward += a.load * self.args.load_multiplier
                elif np.linalg.norm(self.agent_poses[:2, a.index] - [0, 0]) <= self.args.zone1_radius:
                    if self.zone1_load > a.torque:              
                        a.load = a.torque
                        self.zone1_load -= a.torque
                    else:
                        a.load = self.zone1_load
                        self.zone1_load = 0
                    reward += a.load * self.args.load_multiplier
        return reward

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

    def get_observation_space(self):
        return self.observation_space

    def get_action_space(self):
        return self.action_space
