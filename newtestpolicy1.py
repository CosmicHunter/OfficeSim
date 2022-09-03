import torch
import torch.nn as nn
import numpy as np
import itertools
import logging
from policy import Policy
from state import ObservableState, FullState
from pipcheck_modified import *
from scipy.spatial import ConvexHull

import time

from action import *
from graph_network import GraphConvNet


"""Changes made for graph model
in configure change the model , 
transform function is commented previous one ,
predict action from state model function has commented changed code
revert these if u dont want the graph model
"""

def mlp2(input_dim , mlp_dims , last_tanh = True):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2:
            layers.append(nn.ReLU())
        else:
            layers.append(nn.Tanh())    
    net = nn.Sequential(*layers)
    return net

def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    print("model inside ",net)
    return net



activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

class StateActionNetwork(nn.Module):
    def __init__(self,input_dim,mlp_dims):
        super().__init__()
        self.state_action_network = mlp2(input_dim , mlp_dims)

    def forward(self , state):
        action = self.state_action_network(state)
        return action

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, mlp_dims):
        super().__init__()
        print("NORMAL ANN AAYA H #############")
        self.value_network = mlp(input_dim, mlp_dims)

    def forward(self, state):
        # print("value forwards @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" , state.size())
        # print("state ::::: ",state)
        value = self.value_network(state)
        print("hoook from value_network.0:", activation['value_network.0'])
        print("hoook from value_network.1:", activation['value_network.1'])
        print("hoook from value_network.2:", activation['value_network.2'])
        print("hoook from value_network.3:", activation['value_network.3'])
        print("hoook from value_network.4:", activation['value_network.4'])
        print("hoook from value_network.5:", activation['value_network.5'])
        # print("value :::::::::::::::::::: ",value)
        return value

# Neural Network accepts transformed rotated state , that is of 13 dimension , transformed and rotated state.
# cadrl stores this state only in last state
# but orca returns joint state
class newtestpolicy1(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'newtestpolicy1'
        self.trainable = True
        self.multiagent_training = True
        self.kinematics = None
        self.epsilon = None
        self.gamma = None
        self.sampling = None
        self.speed_samples = None
        self.rotation_samples = None
        self.query_env = None
        self.action_space = None
        self.speeds = None
        self.rotations = None
        self.action_values = None
        self.self_state_dim_rot = 6  # in rotated state
        self.other_agent_state_dim_rot = 8 # in rotate state
        self.wall_state_idx_in_rot = 14 # in rotated state , starting idx of wall state
        self.wall_state_dim = 9
        self.static_obs_state_dim = 9
        self.other_agents = None
        self.group_agents = None
        self.num_agents = None
        self.joint_state_dim = self.self_state_dim_rot +  self.other_agent_state_dim_rot + self.wall_state_dim + self.static_obs_state_dim
        self.il_policy_name = 'spf'
        # self.il_policy = policy_factory[self.il_policy_name]()
        
     
    def configure(self, config):

        print("config inside configure function of new test policy" ,config)
        self.set_common_parameters(config)
        mlp_dims = [int(x) for x in config.get('spfrl', 'mlp_dims').split(', ')]
        # Value network is simple mlp
        self.model = ValueNetwork((self.other_agents-1)*self.other_agent_state_dim_rot +  self.joint_state_dim, mlp_dims)
        for name, layer in self.model.named_modules():
            print(name , layer)
            layer.register_forward_hook(get_activation(name))
        
        # uncomment for loading the network of state action
        # self.model = StateActionNetwork((self.other_agents-1)*self.other_agent_state_dim_rot +  self.joint_state_dim, mlp_dims)
        
        # uncomment for graph model
        # if we want the graph neural network
        # print("graph model for state action direct policy learning !!")
        # self.model = GraphConvNet(self.self_state_dim_rot, self.other_agent_state_dim_rot , self.wall_state_dim , self.static_obs_state_dim , 32 , mlp_dims)
        ####################
        self.multiagent_training = config.getboolean('spfrl', 'multiagent_training')
        logging.info('Policy: SPFRL without occupancy map')


    def configure_other_params(self  , num_agents):
        self.num_agents = num_agents
        self.other_agents = num_agents-1
        self.group_agents = self.other_agents
        print(f"number of agents inside newtestpolicy1 is {self.num_agents}")
    
    def set_common_parameters(self, config):
        self.gamma = config.getfloat('rl', 'gamma')
        self.kinematics = config.get('action_space', 'kinematics')
        self.sampling = config.get('action_space', 'sampling')
        self.speed_samples = config.getint('action_space', 'speed_samples')
        self.rotation_samples = config.getint('action_space', 'rotation_samples')
        self.query_env = config.getboolean('action_space', 'query_env')
       
    def set_device(self, device):
        self.device = device
        self.model.to(device)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def build_action_space(self, v_pref):
        """
        Action space consists of 25 uniformly sampled actions in permitted range and 25 randomly sampled actions.
        """
        holonomic = True if self.kinematics == 'holonomic' else False
        speeds = [(np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * v_pref for i in range(self.speed_samples)]
        if holonomic:
            rotations = np.linspace(0, 2 * np.pi, self.rotation_samples, endpoint=False)
        else:
            rotations = np.linspace(-np.pi / 4, np.pi / 4, self.rotation_samples)

        action_space = [ActionXY(0, 0) if holonomic else ActionRot(0, 0)]
        for rotation, speed in itertools.product(rotations, speeds):
            if holonomic:
                action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
            else:
                action_space.append(ActionRot(speed, rotation))

        self.speeds = speeds
        self.rotations = rotations
        self.action_space = action_space

    def propagate(self, state, action):
        if isinstance(state, ObservableState):
            # propagate state of humans
            next_px = state.px + action.vx * self.time_step
            next_py = state.py + action.vy * self.time_step
            next_state = ObservableState(next_px, next_py, action.vx, action.vy, state.radius , state.grp_id)
        elif isinstance(state, FullState):
            # propagate state of current agent
            # perform action without rotation
            if self.kinematics == 'holonomic':
                next_px = state.px + action.vx * self.time_step
                next_py = state.py + action.vy * self.time_step
                next_state = FullState(next_px, next_py, action.vx, action.vy, state.radius,
                                       state.gx, state.gy, state.v_pref ,state.grp_id)
            else:
                next_theta = state.theta + action.r
                next_vx = action.v * np.cos(next_theta)
                next_vy = action.v * np.sin(next_theta)
                next_px = state.px + next_vx * self.time_step
                next_py = state.py + next_vy * self.time_step
                next_state = FullState(next_px, next_py, next_vx, next_vy, state.radius, state.gx, state.gy,
                                       state.v_pref, next_theta ,state.grp_id)
        else:
            raise ValueError('Type error')

        return next_state

    def predict_action_from_state_action_model(self , state , agent_idx):
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')

        if self.reach_destination(state):
            return ActionXY(0 , 0)

        # for normal state action model
        # new_state = self.transform(state).unsqueeze(0)
        # # print(new_state.unsqueeze(0))
        # new_state_norm = nn.functional.normalize(new_state, p=2.0, dim=1, eps=1e-12, out=None)
        # action = self.model(new_state_norm)
        # print("action from policy state action model : ",action)
        
        ##########for graph model

        new_state = self.transform(state)
        print("transformed state size : " , new_state.size())
        
        action = self.model(new_state.unsqueeze(dim = 0))

        print("action from policy graph model : ",action)
        #######################

        return action

    def predict(self, state , agent_idx):
        """
        Input state is the joint state of robot concatenated by the observable state of other agents

        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed

        """
        # print(f"Predict is called for agent idx {agent_idx}")
        t1 = time.time()
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            # print(state)
            return ActionXY(0, 0) 
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)
        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            self.action_values = list()
            max_value = float('-inf')
            max_action = None
            lc = 0
            t1 = time.time()
            for action in self.action_space:
                lc+=1
                next_self_state = self.propagate(state.self_state, action)
                t_one_s = time.time()
                actions_list = [None] * self.num_agents
                actions_list[agent_idx] = action
                ob, reward, done, info,_ = self.env.one_step_lookahead(actions_list , agent_idx)
                t_one_e = time.time()
                # to train without spf reward we have commented it out
                t_rew_s = time.time()
                
                t_rew_e = time.time()
               
                # batch_next_states = torch.cat([torch.Tensor([next_self_state + next_human_state + ob[1][0] + ob[1][1]]).to(self.device)  for next_human_state in ob[0]], dim=0)  
                temp_state_list = []
                for next_human_state in ob[agent_idx][0]:
                    # print(type(next_human_state))
                    temp1 = next_self_state + next_human_state
                    temp2 = ob[agent_idx][1][0] + ob[agent_idx][1][1]
                    temp3 = temp1 + temp2
                    temp_state_list.append(torch.Tensor([temp3]).to(self.device))
                batch_next_states = torch.cat(temp_state_list , dim = 0)
                # size of batch_next_states  is [num agents - 1 , 16]
                # print("bns size " , batch_next_states.size())
                rotated_batch_input = self.rotate(batch_next_states)
                # print(rotated_batch_input.size())
                # size of rotated_batch_input is [3,13]
                rotated_transformed_batch_input = self.transform_rotated_state_to_model_format(rotated_batch_input)
                # print(rotated_transformed_batch_input.size())
                rotated_transformed_batch_input = nn.functional.normalize(rotated_transformed_batch_input, p=2.0, dim=1, eps=1e-12, out=None)
                t_model_s = time.time()
                next_state_value = self.model(rotated_transformed_batch_input).data.item()
                    
                value = reward[agent_idx] + pow(self.gamma, self.time_step * state.self_state.v_pref) * next_state_value
                t_model_e = time.time()
                self.action_values.append(value)
                if value > max_value:
                    # print("control detection ###################")
                    max_value = value
                    max_action = action
                
                print("----------------------------------------")
                print("one step lookahead time " , t_one_e - t_one_s)
                print("model preds time ",t_model_e-t_model_s)
                
        if self.phase == 'train' or self.phase == 'val':
            self.last_state[agent_idx] = self.transform(state)    
        self.last_state_normal[agent_idx] = state
        t2 = time.time()
       
        print("newtestpolicy1 predict total time"  , t2-t1)
        print("----------------------------------------")
        return max_action


    # this transform function takes the state and transfroms it to agent 0 centric state
    #  first converts it to [# no of agents -1 , joint agent centric state lenght] size tensor
    #  This state can be further reduced by removing the values for agent 0 state from other places.

    def transform_rotated_state_to_model_format(self , rotated_state):
        tensor_list = []
        tensor_list.append(rotated_state[0 , :self.wall_state_idx_in_rot])
        static_wall_and_obs_states = rotated_state[0 , self.wall_state_idx_in_rot :]
        for i in range(1 , self.other_agents):
            # print(rotated_state[i])
            tensor_list.append(rotated_state[i , self.self_state_dim_rot:self.wall_state_idx_in_rot])

        tensor_list.append(static_wall_and_obs_states)
        # new state removes the redundant values of first agent that occur at the beggining for [numagents-1 , 13] size tensor
        new_state = torch.cat(tensor_list , dim = 0).to(self.device) # 
        return new_state.unsqueeze(dim = 0)    # shape  size tensor is returned
    

    #######################################################################
    """
    This transform is for graph network only
    """

    # def transform(self, state):    
    #     """
    #     Take the state passed from agent and transform it to tensor for batch training

    #     :param state:
    #     :return: tensor of shape (len(state), )
    #     """
           
    #     # state.self_state is full state of robot
    #     # state.human_state is a list containing observable state of humans
    #     # state.human_state[0] is the first human observable state
    #     # state.self_state + state.human_states[0] gives the concatenated state
       
    #     # print("1 ",state.self_state , type(state.self_state))
    #     # print("2 ",state.wall_state , type(state.wall_state))
    #     # print("3 ",state.static_obstacle_state)
    #     # for human_state in state.other_states:
    #     #     print("4 " , human_state , type(human_state))
    #     # print("5 " , state.wall_state + state.static_obstacle_state)
    #     # a = state.self_state + state.wall_state
    #     # b = state.wall_state + state.self_state

    #     # print("6 ",a)
    #     # print("7 ", type(a))
    #     # print("8" , b)
    #     # print("a + b ", a+b )
    #     temp_state_list = []
    #     for human_state in state.other_states:
    #         temp1 = state.self_state + human_state 
    #         temp2 = state.wall_state + state.static_obstacle_state
    #         temp3 = torch.Tensor([temp1 + temp2])
    #         temp_state_list.append(temp3)
    #     state_tensor = torch.cat(temp_state_list , dim = 0)  
    #     # print("len temp state list " , len(temp_state_list))
    #     # print(state_tensor.size())  
    #     # print(state_tensor)
    #     # state_tensor = torch.cat([torch.Tensor([state.self_state + human_state + state.wall_state + state.static_obstacle_state]).to(self.device)
    #                               # for human_state in state.other_states], dim=0)
    #     # num human , state dimension for 2 agent.
    #     state = self.rotate(state_tensor)   # it gives 31 dimension tensor
        
    #     return state   # of human , state len]

    #######################################################################
    def transform(self, state):    
        """
        Take the state passed from agent and transform it to tensor for batch training

        :param state:
        :return: tensor of shape (len(state), )
        """
           
        # state.self_state is full state of robot
        # state.human_state is a list containing observable state of humans
        # state.human_state[0] is the first human observable state
        # state.self_state + state.human_states[0] gives the concatenated state
       
        temp_state_list = []
        for human_state in state.other_states:
            temp1 = state.self_state + human_state 
            temp2 = state.wall_state + state.static_obstacle_state
            temp3 = torch.Tensor([temp1 + temp2])
            temp_state_list.append(temp3)
        state_tensor = torch.cat(temp_state_list , dim = 0)  
        # state_tensor = torch.cat([torch.Tensor([state.self_state + human_state + state.wall_state + state.static_obstacle_state]).to(self.device)
                                  # for human_state in state.other_states], dim=0)
        # num human , state dimension for 2 agent.
        state = self.rotate(state_tensor)   # it gives 31 dimension tensor
        
        tensor_list = []
        tensor_list.append(state[0 , :self.wall_state_idx_in_rot])
        static_wall_and_obs_states = state[0 , self.wall_state_idx_in_rot :]
        for i in range(1 , self.other_agents):
            
            tensor_list.append(state[i , self.self_state_dim_rot: 14])

        tensor_list.append(static_wall_and_obs_states)

        # new state removes the redundant values of first agent that occur at the beggining for [numagents-1 , 13] size tensor
        new_state = torch.cat(tensor_list , dim = 0).to(self.device) # dimension tensor [self state dim + num agents * other state dim + wall state dim  + static obs state dim]
        # print(new_state.size())

        # so the new state contains the flattened state , with static wall and obs state appened in the end
        # print("transformation state size " , state.size()) 
        return new_state   # of human , state len]

    def rotate(self, state):
        """
        Transform the coordinate to agent-centric.
        Input state tensor is of size (batch_size, state_length)
        

        # This is before rotating .. 

        """
        # state lenght dimension before rotating = 34

        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'grp_id', 'px1', 'py1', 'vx1', 'vy1', 'radius1' ,'grp id2'
        #  0     1      2     3      4        5     6      7         8       9       10     11     12    13         14       
        
        # wall state
        # 'tl[0]' , 'tl[1]' , 'tr[0]' , 'tr[1]' , 'bl[0]' , 'bl[1]',' br[0]' , 'br[1]', closedist
        #   15        16         17        18       19         20       21        22      23

        # obs state
        # 'tl[0]' , 'tl[1]' , 'tr[0]' , 'tr[1]' , 'bl[0]' , 'bl[1]',' br[0]' , 'br[1]', closedist
        #   24        25         26        27       28         29       30        31       32

        batch = state.shape[0]
        # print("in rotate : state shape ",state.shape)
        dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
        dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
        rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        v_pref = state[:, 7].reshape((batch, -1))
        vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
        vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

        radius = state[:, 4].reshape((batch, -1))
        # if self.kinematics == 'unicycle':
        #     theta = (state[:, 8] - rot).reshape((batch, -1))
        # else:
        #     # set theta to be zero since it's not used
        #     theta = torch.zeros_like(v_pref)
        vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
        vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
        px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
        px1 = px1.reshape((batch, -1))
        py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
        py1 = py1.reshape((batch, -1))
        radius1 = state[:, 13].reshape((batch, -1))
        radius_sum = radius + radius1
        da = torch.norm(torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                                  reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)
        gid1 = state[: , 8].reshape((batch , -1))
        # print("group 1 tensor",gid1)
        gid2 = state[: , 14].reshape((batch , -1))
        # print("group 2 tensor" , gid2)

        # wall state cordinates agent centric conversion
        tl_w_0 = ((state[:, 15] - state[:, 0]) * torch.cos(rot) + (state[:,16] - state[:,1]) * torch.sin(rot)).reshape((batch, -1))
        tl_w_1 = ((state[:, 16] - state[:, 1]) * torch.cos(rot) - (state[:,15] - state[:,0]) * torch.sin(rot)).reshape((batch, -1))
       
        tr_w_0 = ((state[:, 17] - state[:, 0]) * torch.cos(rot) + (state[:,18] - state[:,1]) * torch.sin(rot)).reshape((batch, -1))
        tr_w_1 = ((state[:, 18] - state[:, 1]) * torch.cos(rot) - (state[:,17] - state[:,0]) * torch.sin(rot)).reshape((batch, -1))
        
        bl_w_0 = ((state[:, 19] - state[:, 0]) * torch.cos(rot) + (state[:,20] - state[:,1]) * torch.sin(rot)).reshape((batch, -1))
        bl_w_1 = ((state[:, 20] - state[:, 1]) * torch.cos(rot) - (state[:,19] - state[:,0]) * torch.sin(rot)).reshape((batch, -1))
         
        br_w_0 = ((state[:, 21] - state[:, 0]) * torch.cos(rot) + (state[:,22] - state[:,1]) * torch.sin(rot)).reshape((batch, -1))
        br_w_1 = ((state[:, 22] - state[:, 1]) * torch.cos(rot) - (state[:,21] - state[:,0]) * torch.sin(rot)).reshape((batch, -1))
        
        closest_dist_wall = (state[:,23]).reshape((batch , -1))
        
        # obstacle state cordinates agent centric conversion
        tl_obs_0 = ((state[:, 24] - state[:, 0]) * torch.cos(rot) + (state[:,25] - state[:,1]) * torch.sin(rot)).reshape((batch, -1))
        tl_obs_1 = ((state[:, 25] - state[:, 1]) * torch.cos(rot) - (state[:,24] - state[:,0]) * torch.sin(rot)).reshape((batch, -1))
       
        tr_obs_0 = ((state[:, 26] - state[:, 0]) * torch.cos(rot) + (state[:,27] - state[:,1]) * torch.sin(rot)).reshape((batch, -1))
        tr_obs_1 = ((state[:, 27] - state[:, 1]) * torch.cos(rot) - (state[:,26] - state[:,0]) * torch.sin(rot)).reshape((batch, -1))
       
        bl_obs_0 = ((state[:, 28] - state[:, 0]) * torch.cos(rot) + (state[:,29] - state[:,1]) * torch.sin(rot)).reshape((batch, -1))
        bl_obs_1 = ((state[:, 29] - state[:, 1]) * torch.cos(rot) - (state[:,28] - state[:,0]) * torch.sin(rot)).reshape((batch, -1))
         
        
        br_obs_0 = ((state[:, 30] - state[:, 0]) * torch.cos(rot) + (state[:,31] - state[:,1]) * torch.sin(rot)).reshape((batch, -1))
        br_obs_1 = ((state[:, 31] - state[:, 1]) * torch.cos(rot) - (state[:,30] - state[:,0]) * torch.sin(rot)).reshape((batch, -1))
      
        
        closest_dist_obstacle = (state[:,32]).reshape((batch , -1))


        new_state = torch.cat([dg, v_pref,radius, vx, vy, gid1,px1, py1, vx1, vy1, radius1, da, radius_sum , gid2,\
                               tl_w_0,tl_w_1 , tr_w_0,tr_w_1,bl_w_0,bl_w_1,br_w_0,br_w_1,closest_dist_wall,\
                               tl_obs_0 , tl_obs_1 , tr_obs_0,tr_obs_1 ,bl_obs_0,bl_obs_1 ,br_obs_0,br_obs_1,closest_dist_obstacle], dim=1)
        # print("new_state tensor" ,new_state)
        return new_state


# new state after rotation
# dg, v_pref, radius, vx, vy, gid1,px1, py1, vx1, vy1, radius1, da, radius_sum , gid2 , 9 wall states , 9 obstacle states
# 0     1       2      3   4    5    6   7    8    9    10      11     12         13           
# dg, v_pref, theta, radius, vx, vy, gid1,px1, py1, vx1, vy1, radius1, da, radius_sum , gid2
# dg, v_pref, theta, radius, vx, vy, gid1,px2, py2, vx2, vy2, radius1, da, radius_sum , gid2

# after rotating states is [13 + 18] = 31 dimension
# self state dimension in rotated state = 6
# other agent state dimension in rotated state = 8

# agent of 31 dimension [agent0 , agent1 , wall ,static] 