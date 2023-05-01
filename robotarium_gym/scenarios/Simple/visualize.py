from rps.utilities.misc import *
from robotarium_gym.scenarios.base import BaseVisualization

class Visualize(BaseVisualization):
    def __init__(self, args):
        self.agent_marker_size_m = .07
        self.goal_marker_size_m = .05
        self.line_width = 1
        self.CM = plt.cm.get_cmap('hsv', 7) # Agent/goal color scheme
        self.show_figure = True
    
    def initialize_markers(self, robotarium, agents):
        '''
        Initializes the marker for agents
        '''
        marker_size_agent = determine_marker_size(robotarium, self.agent_marker_size_m)
        marker_size_goal = determine_marker_size(robotarium,self.goal_marker_size_m)          

        self.agent_markers = [robotarium.axes.scatter( \
                agents.agent_poses[0,ii], agents.agent_poses[1,ii], \
                s=marker_size_agent, marker='.', facecolors=self.CM(2), 
                edgecolors=self.CM(2), linewidth=self.line_width, zorder=-2) for ii in range(agents.num_agent)]
        
        self.goal_markers = [robotarium.axes.scatter( \
                agents.goal_loc[ii][0], agents.goal_loc[ii][1], \
                s=marker_size_goal, marker='.', facecolors=self.CM(2), 
                edgecolors=self.CM(2), linewidth=self.line_width, zorder=-2) for ii in range(agents.num_goal)]
    
    def update_markers(self, robotarium, agents ):

        '''
        Update visualization for the agents and goal for each frame
        '''
        
        for i in range(agents.agent_poses.shape[1]):
            self.agent_markers[i].set_offsets(agents.agent_poses[:2,i].T)
            self.agent_markers[i].set_sizes([determine_marker_size(robotarium, self.agent_marker_size_m)])
        
        