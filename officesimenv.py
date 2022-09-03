import gym
from gym import spaces
from ast import AST
from operator import ge, truediv
from shutil import move
from turtle import pos
import pygame
import math
import random
import sys
import time 
import numpy as np
from helper_for_testsim import *
from astar_planner import *
from helper_classes import *
from data_points import *
from helper_functions import *
from state import *
from action import *
import logging
from info import *

WHITE = (255,255,255)
isRunning = True
cmap = {"Green" : (0,255,0) ,
        "Red" : (255 , 0 , 0)}



# Move this inside render only

# add wall collision metric
# plot the plots
# check garbage in rewards
# module taking most time

       
wall_list = []
office_list = []
gate_list = []
bc_list = []
grp_id = 0

for idx , item in office_corner_cordinate_dict.items():
    office_list.append(Office(idx , item[0],item[1],item[2], item[3]))

for idx , item in wall_end_point_cordinate_dict.items():
    wall_list.append(Wall(idx , item[0][0],item[0][1],item[1][0],item[1][1]))

for idx , item in gate_end_point_cordinate_dict.items():
    gate_list.append(Gate(idx , item[0][0],item[0][1],item[1][0],item[1][1],item[2]))

# Box cell 1 is corridor in the office
bc1 = BoxCell(0,(45,478),(945,478), (45,561) ,(945,561))
bc_list.append(bc1)





def read_marker_file(file):
   
    bpt_list = []
    try:
        with open(file , "r") as f:
            for line in f.readlines():
                line = line.strip()
                bm , mx , my = line.split(" ")
                
                bpt_list.append((bm , mx , my))
        return bpt_list
       
    except: 
        print("Error encountered in reading marker file")

def read_graph_points_file(file):
   
    graph_points = []
    try:
        with open(file , "r") as f:
            for line in f.readlines():
                line = line.strip()
                mx , my = line.split(" ")
                graph_points.append((int(mx) , int(my)))
        return graph_points
       
    except: 
        print("Error encountered in reading graph points file")

def create_graph_nodes_dict(graph_dict):    
    graph_node_dict = {}
    node_list = []
    for k in graph_dict.keys():
        node_list.append(AStar_Node((k[0],k[1]),0))
    for k , v in graph_dict.items():
        node1 = get_node_by_cordinate((k[0],k[1]),node_list)
        val_list = []
        for pt in v:
            temp = get_node_by_cordinate((pt[0],pt[1]),node_list)
            val_list.append(temp)
        graph_node_dict[node1] = val_list
    return graph_node_dict

def get_path_astar(spt , ept , graph_dict):
    # print("astar block started")
    # spt = (241,518)
    # ept = (825,220)
    
    snode = None
    enode = None
    graph_node_dict = create_graph_nodes_dict(graph_dict)
    for key in graph_node_dict.keys():
        if key.pos == spt:
            snode = key
        if key.pos == ept:
            enode = key

    path = astar(snode , enode , graph_node_dict)
    # print("PATH  ======>",path)
    # print("astar block ended")
    return path




def draw_global(screen,bg_img ,font, boundary_points,graph_points ,agent_list, goal_list , force_angle_list = None):
    screen.blit(bg_img , (0,0))
    for item in boundary_points:
        pygame.draw.circle(screen, cmap[item[0]], (int(item[1]),int(item[2])), 5, 2)
    
    for item in graph_points:
        pygame.draw.circle(screen ,(255,192 , 203) ,(int(item[0]),int(item[1])) , 5 , 2)
    for goal in goal_list:
        goal.draw_goal(screen)
    for agent in agent_list:
        agent.draw_agent(screen, font)
    if force_angle_list != None:
        for idx , angle in enumerate(force_angle_list):
            agent_list[idx].draw_arrow(screen , angle)
    # Show graph map
    for k , v in graph_dict.items():
        for pt in v:
            pygame.draw.line(screen , (0,0,255), k , pt)

    pygame.display.update()

def draw_global_v2(screen,bg_img , font,boundary_points,graph_points ,agent_list ,agent_points, goal_list, force_angle_list = None):
    screen.blit(bg_img , (0,0))
    for item in boundary_points:
        pygame.draw.circle(screen, cmap[item[0]], (int(item[1]),int(item[2])), 5 , 2)
    
    for item in graph_points:
        pygame.draw.circle(screen ,(255,192 , 203) ,(int(item[0]),int(item[1])) , 5 , 2)
    for goal in goal_list:
        goal.draw_goal(screen)
    for idx , agent_point in enumerate(agent_points):
        agent = agent_list[idx]
        agent.draw_agent(screen , font,agent_point)
    if force_angle_list != None:
        for idx , angle in enumerate(force_angle_list):
            agent_list[idx].draw_arrow(screen , angle , agent_points[idx])
    # Show graph map
    for k , v in graph_dict.items():
        for pt in v:
            pygame.draw.line(screen , (0,0,255), k , pt)

    pygame.display.update()


# update info function to return multiple infos
def get_agent_info(agent , agent_list , rect_obstacle_list ,rect_wall_list):
    reward = 0 
    info = None
    dmin = float('inf')
    # goal reach condition not working 
    # print("get agent info function :==========> " , agent.is_final_goal_reached)
    for a in agent_list:
        dmin = float('inf')
        if  a!= agent:
            d = np.sqrt((a.x - agent.x)**2 + (a.y - agent.y)**2)
            # print(d)
            if d < 2 * a.radius:
                info = "collision"
                break
            elif d < dmin:
                dmin = d
                info = None

    if agent.is_final_goal_reached == True:
        info = "goalreached"
        
    if agent.is_final_goal_reached == False:
        agent_office = chk_agent_in_which_office(office_list , agent.x , agent.y)
        goal_office = chk_agent_in_which_office(office_list , agent.goal.x , agent.goal.y)
        if agent_office == goal_office:
            info = "officereached"
   
                
    for obs in rect_obstacle_list:
        if chk_circle_rect_collision(agent.getPos() ,agent.radius, obs):
            info = "collision"
            break

    for wall in rect_wall_list:
        if chk_circle_rect_collision(agent.getPos() , agent.radius , wall):
            info = "wallcollision"
            break

    # print("returning info -----------> " , info)
    # print("returning  dmin val -------->" ,dmin)
    return info , dmin


class OfficeEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        super(OfficeEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        # self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input (channel-first; channel-last also works):
        # self.observation_space = spaces.Box(low=0, high=255,
                                            # shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)
        self.config = None
        self.agent_list = []
        self.goal_list = []
        self.FPS = 60
        self.WIDTH , self.HEIGHT = 992, 892
        # self.screen = pygame.display.set_mode((self.WIDTH , self.HEIGHT))
        # self.screen.fill(WHITE)
        # self.bg_img = pygame.image.load("officefloormap.png")
        # self.clock_obj = pygame.time.Clock()
        self.screen = None
        self.bg_img = None
        self.clock_obj = None
        self.font = None
        self.agent_idx = 0
        self.rect_obstacle_list = generate_rect_obstacles(obstacle_list)
        self.boundary_point_file = "boundary_marker2.txt"
        self.graph_point_file = "graph_nodes.txt"
        self.boundary_points = read_marker_file(self.boundary_point_file)
        self.graph_points =   read_graph_points_file(self.graph_point_file)
        self.simulation_step  = 0
        # self.max_steps = 500 # let them reduce to less than 300
        # for traj pred gen data task
        self.max_steps = 320
        ##########
        self.goal_reach_reward = 1
        self.office_reach_reward = 0.4
        # 2 times the euclidean distance.
        self.max_steps_elapsed_reward = -0.008
        self.collision_reward = 0
        self.wall_collision_reward = -0.18
        self.rect_wall_list = generate_rect_walls(wall_list)
        self.discomfort_dist = 0
        self.discomfort_penalty_factor = 0

        self.time_step = None
        self.global_time = None
        self.time_limit = None
        # for wall in self.rect_wall_list:
            # print(f"Rect Wall , left : {wall.left} , top :{wall.top} , w : {wall.w} , h :{wall.h}")
        self.render_states = list()
        self.case_capacity = None
        self.case_size = None
        self.train_val_sim = None
        self.test_sim = None
        self.case_counter = None

    
    def configure(self, config):
        self.config = config
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        # self.randomize_attributes = config.getboolean('env', 'randomize_attributes')
        # self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_reward = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        # self.group_success_reward = config.getfloat('reward' , 'group_success_reward')
        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
                          'test': config.getint('env', 'test_size')}
        self.train_val_sim = config.get('sim', 'train_val_sim')
        self.test_sim = config.get('sim', 'test_sim')
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        # logging.info('num agents: {}'.format(self.num_agents))
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        
    def temp_chk(self):
        for agent in self.agent_list:
            if check_if_pos_safe_v2(agent.getPos() , self.agent_list , self.rect_obstacle_list , self.rect_wall_list)==False:
                    return False
        return True
    def step(self , actions = None , update_pos = True , render = False):
        if update_pos:
            self.simulation_step+=1

            # print("step : ",self.simulation_step)
        # if actions == None:
        fa =[] 
        for i in range(len(self.agent_list)):
            # step if no action provided ( using internal policy (i.e spf))
            # print("index " , i)
            if actions[i] == None:
                fl_attr, fl_rep , temp = self.agent_list[i].move_agent(self.agent_list,wall_list,bc_list,self.rect_obstacle_list)
                if self.agent_list[i].is_step_safe(self.agent_list,self.rect_obstacle_list,self.rect_wall_list) and update_pos:
                    self.agent_list[i].update()
            
                case , oid = self.agent_list[i].determine_entering_exiting(graph_point_inside_office_dict , graph_point_infront_office_dict)
            
            
                if case == "entering" and oid !=-1:
                    if self.agent_list[i].is_safe_to_enter(self.agent_list ,gate_list[oid],graph_point_inside_office_dict , graph_point_infront_office_dict) == False:
                        self.agent_list[i].waiting_mode = True
                        # print(f"waiting mode on for agent idx {i}")
                    else:
                        self.agent_list[i].waiting_mode = False    
                fa.append(temp)

            # step if action is provided
            # action should be list of actions for each agent
            elif actions[i] != None:
                temp = self.agent_list[i].move_agent_using_action(actions[i])
                if self.agent_list[i].is_step_safe(self.agent_list,self.rect_obstacle_list,self.rect_wall_list) and update_pos:
                    self.agent_list[i].update()
                fa.append(temp)    
        agent_points = [agent.getPos() for agent in self.agent_list]
        render_state = [agent_points , fa]
        self.render_states.append(render_state)
        if render:
            draw_global(self.screen , self.bg_img ,self.font, self.boundary_points,self.graph_points , self.agent_list , self.goal_list , fa)
       
        # observation is a observation list that gets observation for all the agents
        observation = []
        for agent in self.agent_list:
            ob = [a.get_observable_state() for a in self.agent_list if a!=agent]
            # new obs = [[ob] , [wall state , obs state]] 
            added_obs = [agent.get_wall_state(self.rect_wall_list) , agent.get_static_obs_state(self.rect_obstacle_list)]
            newob = [ob , added_obs]
            observation.append(newob)
        
        # the step function must check the condition of simulation and return the proper reward for taking that step
        # the step function must return the info also


        # reward is agent reaching office less penalty
        # best case agent reach the correct office + goal
        # agent not reaching office timeout penalty
        # agent colliding .collision penalty

        # is goal reached + is office reached
        # ispe reward aajaega
        # when all are reached or max steps elapsed simulation is done
        dones = []
        infos = []
        rewards = []
        spf_rew_data = []
        if update_pos:
            self.global_time += self.time_step


        """
        ADD ROBOT FIXED DONE SIGNAL , SO ONCE A ROBOT HAS REACHED GOAL ONLY ONE TIME IT GETS THAT REWARD
        """
        for idx , agent in enumerate(self.agent_list):
            spfreward, flag = spf_reward(agent.last_state)
            # print(f"agent idx {idx} , agent.fixed_done_signal : {agent.fixed_done_signal}")
            if agent.fixed_done_signal == None:
                if self.simulation_step >= self.max_steps:
                    # print("1")
                    done = True 
                    info = "timeout"
                    reward = self.max_steps_elapsed_reward    
                    agent.fixed_done_signal = True 
                    agent.fixed_info_signal = info        
                else:
                    info , dmin = get_agent_info(agent , self.agent_list ,self.rect_obstacle_list , self.rect_wall_list)
                    if info == "goalreached":
                        reward = self.goal_reach_reward
                        done = True
                        agent.fixed_done_signal = True 
                        agent.fixed_info_signal = info
                    elif info == "officereached": # add one time
                        if agent.office_reach_flag == False:
                            reward = self.office_reach_reward
                            agent.office_reach_flag = True
                        else:
                            reward = -0.007
                        done = False
                        
                    elif info == "collision":

                        reward = self.collision_reward
                        done = True
                        agent.fixed_done_signal = True 
                        agent.fixed_info_signal = info

                    elif info == "wallcollision":
                        reward = self.wall_collision_reward
                        done = False

                    elif dmin < self.discomfort_dist:
                        reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
                        done = False
                        if info == None:
                            info = Danger(dmin)
                    
                    else:
                        reward = -0.007
                        done = False
                        if info == None:
                            info = "Nothing"

            else:
                reward = 0
                done = agent.fixed_done_signal
                info = agent.fixed_info_signal    
                
            # spfreward, flag = spf_reward(agent.last_state) 
            avoidgroupbreakreward = avoid_group_break_reward(agent.last_state)
            # print(f"pure rew col : {reward}")
            reward = reward + spfreward + avoidgroupbreakreward 
            
            # print(f" Agent idx  :{idx} , spf reward  : {spfreward}")
            
            # print(f"Agent idx : {idx} , avoidgroupbreakreward : {avoidgroupbreakreward}")
            # print("reward : " , reward)
            rewards.append(reward)
            infos.append(info)
            dones.append(done)
            spf_rew_data.append((spfreward ,flag))

        if self.is_there_collision(infos):
            for i in range(len(dones)):
                dones[i] = True   # in order to stop explorer when collision happens
                if infos[i]== "Nothing" or isinstance(infos[i],Danger):
                    infos[i] = GroupCollided()

        
        return observation, rewards, dones, infos, spf_rew_data
    
    def one_step_lookahead(self,actions,agent_idx):
        # modify the step function to incorporate one step lookahead
        return self.step(actions , update_pos = False)
        
    def reset(self , phase='test', test_case=None):
        global grp_id
        self.global_time = 0
        self.simulation_step = 0
        self.agent_list = []
        self.goal_list = []
        self.agent_idx = 0
        self.render_states = list()
        grp_id = 0
        # print(test_case)
        if test_case is not None:
            self.case_counter[phase] = test_case

        # secnario where multiple agents are spawned in 2 offices    
        # self.generate_scenario_v1(0 , 1, office_list,gate_list , 15, 5 , self.graph_points ,graph_dict ,self.rect_obstacle_list)
        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                              'val': 0, 'test': self.case_capacity['val']}
            
        if self.case_counter[phase] >= 0:
            print(counter_offset[phase])
            print(self.case_counter[phase])
            np.random.seed(counter_offset[phase] + self.case_counter[phase])
        # print("case counter : ",counter_offset[phase] + self.case_counter[phase])
        # scenario where there can be groups of agents spawned in various offices
        # scenario can be customized
        self.generate_scenario_v3(self.graph_points,graph_dict,gate_list)
        # self.generate_scenario_v4_fix_src_goal(self.graph_points,graph_dict,gate_list)
        # self.generate_scenario_v3_less_variation(self.graph_points,graph_dict,gate_list)
        self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
        observation = []
        for agent in self.agent_list:
            agent.time_step = self.time_step
        for agent in self.agent_list:
            ob = [a.get_observable_state() for a in self.agent_list if a!=agent]
            # newob = [[ob] , [wall , obstacle]]
            added_obs = [agent.get_wall_state(self.rect_wall_list) , agent.get_static_obs_state(self.rect_obstacle_list)]
            newob = [ob , added_obs]
            observation.append(newob)
        return observation  # reward, done, info can't be included
    
    def render(self, mode='human'):
        # there needs to be a self states list / dict that needs to be passed
        if self.screen == None:

            pygame.init()
            self.font = pygame.font.SysFont(None, 20)
            self.screen = pygame.display.set_mode((self.WIDTH , self.HEIGHT))
            self.screen.fill(WHITE)
            self.bg_img = pygame.image.load("officefloormap.png")
            self.clock_obj = pygame.time.Clock()

        timesteps = len(self.render_states)
        for t in range(timesteps):
            self.clock_obj.tick(self.FPS)
            agent_points = self.render_states[t][0]
            fa = self.render_states[t][1]
            draw_global_v2(self.screen , self.bg_img,self.font , self.boundary_points,self.graph_points , self.agent_list,agent_points ,self.goal_list, fa)
        time.sleep(1)
    
    def close (self):
        pygame.quit()
        sys.exit()
    
    def is_done(self , dones):
        return np.all(dones)

    def is_there_collision(self ,info_list):
        for i in range(len(info_list)):
            if info_list[i] == "collision":
                return True
        return False

    def update_last_state(self , ob_list):
        # after every step we update the last state , this will be helpful for memory storing and 
        # computing group rewards
        for idx , agent in enumerate(self.agent_list):
            state = agent.get_joint_state(ob_list[idx])
            agent.last_state = state

    def generate_scenario_v3(self,graph_points , graph_dict , gate_list):
        global grp_id

        if np.random.random() < 0.5:
            px_noise = np.random.normal(0,1,1)[0] 
            py_noise = np.random.normal(0,1,1)[0]
            xpos1 = np.random.randint(86 , 235) + px_noise
            ypos1 = np.random.randint(608 , 615) + py_noise
        else:
            px_noise = np.random.normal(0,1,1)[0]
            py_noise = np.random.normal(0,1,1)[0]
            xpos1 = np.random.randint(118 , 240) + px_noise
            ypos1 = np.random.randint(508 , 518) + py_noise
        rand_grp_size1 = 1
        # xpos1 , ypos1 , rand_grp_size1 = 171 , 517 , 4
        grp_positions = generate_grp_positions((xpos1,ypos1) , rand_grp_size1)
        rand_color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        goal1 = Goal(214,303 ,rand_color)
        self.goal_list.append(goal1)
        for pos in grp_positions:
                agent = Agent(pos[0], pos[1] , rand_color , self.agent_idx)
                self.agent_idx +=1
                agent.goal = goal1
                agent.grp_id = grp_id
                agent.assign_path_to_agent(self.graph_points , graph_dict)
                agent.curr_subgoal = agent.sgl[0]
                agent.curr_subgoal_idx = 0
                self.agent_list.append(agent)
        grp_id+=1

        px_noise = np.random.normal(0,1,1)[0]
        py_noise = np.random.normal(0,1,1)[0]
        xpos2 = np.random.randint(88 , 186) + px_noise
        ypos2 = np.random.randint(300 , 410) + py_noise
        rand_grp_size2 = 1
        # xpos2 , ypos2 , rand_grp_size2 = 116, 362 , 4
        grp_positions = generate_grp_positions((xpos2,ypos2) , rand_grp_size2)
        rand_color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        goal2 = Goal(205,518 ,rand_color)
        self.goal_list.append(goal2)
        for pos in grp_positions:
                agent = Agent(pos[0], pos[1] , rand_color , self.agent_idx)
                self.agent_idx +=1
                agent.goal = goal2
                agent.grp_id = grp_id
                agent.assign_path_to_agent(self.graph_points , graph_dict)
                agent.curr_subgoal = agent.sgl[0]
                agent.curr_subgoal_idx = 0
                self.agent_list.append(agent)
        grp_id+=1

        for agent in self.agent_list:
            agent.compute_gate_walls(gate_list)

        # print("---------------- test case : info ----------------")
        # for agent in self.agent_list:
        #     print(f"Agent ID {agent.id} , Pos : {agent.getPos()} , Goal : {agent.goal.getPos()}")
    

    # def generate_scenario_v3_less_variation(self,graph_points , graph_dict , gate_list):
    #     global grp_id

    #     # if np.random.random() < 0.5:
    #     px_noise = np.random.normal(0,1,1)[0] 
    #     py_noise = np.random.normal(0,1,1)[0]
    #     xpos1 = np.random.randint(86 , 135) + px_noise
    #     ypos1 = np.random.randint(608 , 615) + py_noise
    #     # else:
    #     #     px_noise = np.random.normal(0,1,1)[0]
    #     #     py_noise = np.random.normal(0,1,1)[0]
    #     #     xpos1 = np.random.randint(118 , 240) + px_noise
    #     #     ypos1 = np.random.randint(508 , 518) + py_noise
    #     rand_grp_size1 = 1
    #     # xpos1 , ypos1 , rand_grp_size1 = 171 , 517 , 4
    #     grp_positions = generate_grp_positions((xpos1,ypos1) , rand_grp_size1)
    #     rand_color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
    #     goal1 = Goal(214,303 ,rand_color)
    #     self.goal_list.append(goal1)
    #     for pos in grp_positions:
    #             agent = Agent(pos[0], pos[1] , rand_color , self.agent_idx)
    #             self.agent_idx +=1
    #             agent.goal = goal1
    #             agent.grp_id = grp_id
    #             agent.assign_path_to_agent(self.graph_points , graph_dict)
    #             agent.curr_subgoal = agent.sgl[0]
    #             agent.curr_subgoal_idx = 0
    #             self.agent_list.append(agent)
    #     grp_id+=1

    #     px_noise = np.random.normal(0,1,1)[0]
    #     py_noise = np.random.normal(0,1,1)[0]
    #     xpos2 = np.random.randint(88 , 126) + px_noise
    #     ypos2 = np.random.randint(350 , 410) + py_noise
    #     rand_grp_size2 = 1
    #     # xpos2 , ypos2 , rand_grp_size2 = 116, 362 , 4
    #     grp_positions = generate_grp_positions((xpos2,ypos2) , rand_grp_size2)
    #     rand_color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
    #     goal2 = Goal(205,518 ,rand_color)
    #     self.goal_list.append(goal2)
    #     for pos in grp_positions:
    #             agent = Agent(pos[0], pos[1] , rand_color , self.agent_idx)
    #             self.agent_idx +=1
    #             agent.goal = goal2
    #             agent.grp_id = grp_id
    #             agent.assign_path_to_agent(self.graph_points , graph_dict)
    #             agent.curr_subgoal = agent.sgl[0]
    #             agent.curr_subgoal_idx = 0
    #             self.agent_list.append(agent)
    #     grp_id+=1

    #     for agent in self.agent_list:
    #         agent.compute_gate_walls(gate_list)

    #     # print("---------------- test case : info ----------------")
    #     # for agent in self.agent_list:
    #     #     print(f"Agent ID {agent.id} , Pos : {agent.getPos()} , Goal : {agent.goal.getPos()}")
    
    def generate_scenario_v3_less_variation(self,graph_points , graph_dict , gate_list):
        # simplified version
        # to be used for traj pred testing
        global grp_id

        # if np.random.random() < 0.5:
        px_noise = np.random.normal(0,1,1)[0] 
        py_noise = np.random.normal(0,1,1)[0]
        xpos1 = np.random.randint(86 , 135) + px_noise
        ypos1 = np.random.randint(608 , 615) + py_noise
        # else:
        #     px_noise = np.random.normal(0,1,1)[0]
        #     py_noise = np.random.normal(0,1,1)[0]
        #     xpos1 = np.random.randint(118 , 240) + px_noise
        #     ypos1 = np.random.randint(508 , 518) + py_noise
        rand_grp_size1 = 1
        # xpos1 , ypos1 , rand_grp_size1 = 171 , 517 , 4
        grp_positions = generate_grp_positions((xpos1,ypos1) , rand_grp_size1)
        rand_color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        goal1 = Goal(112,497 ,rand_color)
        self.goal_list.append(goal1)
        for pos in grp_positions:
                agent = Agent(pos[0], pos[1] , rand_color , self.agent_idx)
                self.agent_idx +=1
                agent.goal = goal1
                agent.grp_id = grp_id
                agent.assign_path_to_agent(self.graph_points , graph_dict)
                agent.curr_subgoal = agent.sgl[0]
                agent.curr_subgoal_idx = 0
                self.agent_list.append(agent)
        grp_id+=1

        px_noise = np.random.normal(0,1,1)[0]
        py_noise = np.random.normal(0,1,1)[0]
        xpos2 = np.random.randint(88 , 126) + px_noise
        ypos2 = np.random.randint(350 , 410) + py_noise
        rand_grp_size2 = 1
        # xpos2 , ypos2 , rand_grp_size2 = 116, 362 , 4
        grp_positions = generate_grp_positions((xpos2,ypos2) , rand_grp_size2)
        rand_color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        goal2 = Goal(240,533 ,rand_color)
        self.goal_list.append(goal2)
        for pos in grp_positions:
                agent = Agent(pos[0], pos[1] , rand_color , self.agent_idx)
                self.agent_idx +=1
                agent.goal = goal2
                agent.grp_id = grp_id
                agent.assign_path_to_agent(self.graph_points , graph_dict)
                agent.curr_subgoal = agent.sgl[0]
                agent.curr_subgoal_idx = 0
                self.agent_list.append(agent)
        grp_id+=1

        for agent in self.agent_list:
            agent.compute_gate_walls(gate_list)

        print("---------------- test case : info ----------------")
        for agent in self.agent_list:
            print(f"Agent ID {agent.id} , Pos : {agent.getPos()} , Goal : {agent.goal.getPos()}")
    

    def generate_scenario_v4_fix_src_goal(self,graph_points , graph_dict , gate_list):
        global grp_id

      
        rand_grp_size1 = 3
        # xpos1 , ypos1 , rand_grp_size1 = 171 , 517 , 4
        xpos1 , ypos1 = 203 , 623
        grp_positions = generate_grp_positions((xpos1,ypos1) , rand_grp_size1)
        rand_color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        # goal1 = Goal(214,303 ,rand_color)
        goal1 = Goal(255 , 256 , rand_color)
        self.goal_list.append(goal1)
        for pos in grp_positions:
                agent = Agent(pos[0], pos[1] , rand_color , self.agent_idx)
                self.agent_idx +=1
                agent.goal = goal1
                agent.grp_id = grp_id
                agent.assign_path_to_agent(self.graph_points , graph_dict)
                agent.curr_subgoal = agent.sgl[0]
                agent.curr_subgoal_idx = 0
                self.agent_list.append(agent)
        grp_id+=1

        rand_grp_size2 = 1
        # xpos2 , ypos2 , rand_grp_size2 = 116, 362 , 4
        xpos2 , ypos2  = 238 , 300
        grp_positions = generate_grp_positions((xpos2,ypos2) , rand_grp_size2)
        rand_color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        # goal2 = Goal(205,518,rand_color)
        goal2 = Goal(111 ,616,rand_color)
        self.goal_list.append(goal2)
        for pos in grp_positions:
                agent = Agent(pos[0], pos[1] , rand_color , self.agent_idx)
                self.agent_idx +=1
                agent.goal = goal2
                agent.grp_id = grp_id
                agent.assign_path_to_agent(self.graph_points , graph_dict)
                agent.curr_subgoal = agent.sgl[0]
                agent.curr_subgoal_idx = 0
                self.agent_list.append(agent)
        grp_id+=1

        for agent in self.agent_list:
            agent.compute_gate_walls(gate_list)

    

    def generate_scenario_v1(self,oid1 , oid2 , office_list,gate_list, nagent1 , nagent2 , graph_points , graph_dict , rect_obstacle_list):
        # here groups are not there
        office1 = office_list[oid1]
        office2 = office_list[oid2]
        global grp_id
        # generate nagent1 points in office1
        rand_color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        # office idx 6
        # goal1 = Goal(834 ,270 , rand_color)
        # office idx 5
        # goal1 = Goal(851 , 411 , rand_color)
        # office idx 4 
        # goal1 = Goal(874 , 606 , rand_color)
        # office idx 3
        # goal1 = Goal(578 , 603 , rand_color)
        # office idx 1
        goal1 = Goal(165 , 614, rand_color)
        self.goal_list.append(goal1)    
        for i in range(nagent1):
            xpos = np.random.randint(office1.tl[0]+9 , office1.tr[0]-9)
            ypos = np.random.randint(office1.tl[1]+9 , office1.bl[1]-9)
            while(check_if_pos_safe((xpos,ypos) , self.agent_list , self.rect_obstacle_list) != True):
                xpos = np.random.randint(office1.tl[0]+9 , office1.tr[0]-9)
                ypos = np.random.randint(office1.tl[1]+9 , office1.bl[1]-9)
            agent = Agent(xpos , ypos , rand_color , self.agent_idx)
            self.agent_idx+=1
            agent.grp_id = grp_id
            grp_id +=1
            agent.goal = goal1
            agent.assign_path_to_agent(self.graph_points , graph_dict)
            agent.curr_subgoal = agent.sgl[0]
            agent.curr_subgoal_idx = 0
            self.agent_list.append(agent)
        

        # generate nagent2 agents in office2
        rand_color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        goal2 = Goal(250,250 ,rand_color)
        self.goal_list.append(goal2)
        for i in range(nagent2):
            while(check_if_pos_safe((xpos,ypos) , self.agent_list , self.rect_obstacle_list) != True):
                xpos = np.random.randint(office2.tl[0]+9 , office2.tr[0]-9)
                ypos = np.random.randint(office2.tl[1]+9 , office2.bl[1]-9)
            
            agent = Agent(xpos , ypos , rand_color , self.agent_idx)
            self.agent_idx+=1
            agent.grp_id = grp_id
            grp_id +=1
            agent.goal = goal2
            agent.assign_path_to_agent(self.graph_points , graph_dict)
            agent.curr_subgoal = agent.sgl[0]
            agent.curr_subgoal_idx = 0
            self.agent_list.append(agent)

        for agent in self.agent_list:
            agent.compute_gate_walls(gate_list)




class Agent:

    def __init__(self,x , y , color , id):
        self.x = x
        self.y = y
        self.virt_x = x
        self.virt_y = y
        self.is_final_goal_reached = False
        self.radius = 9
        self.goal = None
        self.prev = [(self.x,self.y)]
        self.oblist = []
        self.isAgent = True
        self.color = color
        self.vx = 0
        self.vy = 0
        self.time_step = 0
        self.v_pref = 1.5
        self.grp_id = None
        self.sgl = []
        self.curr_subgoal = None
        self.curr_subgoal_idx = -1
        self.id = id
        # self.v_pref = 
        self.last_state = None
        self.grp_agents = []
        self.gate_walls = []
        self.waiting_mode = False # in waiting mode the agent detects its not ok to enter a room at the moment so , its goal shifts to its center temporarily and agents waits 
        # the agents is aware of the forces in this mode , and hopefully will not create a blockade.
        self.fixed_done_signal = None
        self.fixed_info_signal = None
        self.office_reach_flag = False
        self.spf_action = None

    def getPos(self):
        return (self.x,self.y)
    
    def setPos(self, x1 , y1):
        self.x = x1
        self.y = y1
    
    def get_joint_state(self ,newob):
        # given some observation ob
        # returns the joint state for prediction and storing memory
        ob = newob[0]
        wallob = newob[1][0]
        obstacleob = newob[1][1]
        state = JointState(self.get_full_state(), ob , wallob , obstacleob)
        return state

    def act(self , state, policy):
        # state = self.get_joint_state(newob)
        action = policy.predict(state , self.id)
        return action

    def get_action(self , state ,policy):
        action = policy.predict_action_from_state_action_model(state , self.id)
        print(action)
        # print(action.shape)
        # print(float(action[0][0]))
        # print(type(action[0][0]))
        if isinstance(action , ActionXY)== False:
            action = ActionXY(float(action[0][0]),float(action[0][1]))
        return action

    def get_observable_state(self):
        return ObservableState(self.x, self.y, self.vx, self.vy, self.radius ,self.grp_id)
    
    def get_full_state(self):
        return FullState(self.x, self.y, self.vx, self.vy, self.radius, self.goal.x, self.goal.y, self.v_pref, self.grp_id)
    
    def isGoalReached(self):
        if self.curr_subgoal != self.goal.getCenter() and self.curr_subgoal != None:
            cx , cy = self.curr_subgoal
        else:
            cx, cy = self.goal.getCenter()  
        if abs(cx - self.x) < (22 + self.radius) + 2 and abs(cy - self.y) < (22 + self.radius+2):
            self.isReached = True
            # print(self.id,self.curr_subgoal_idx)
            return True

        return False
    
    def get_wall_state(self, rect_wall_list):
        # gets the state of closest wall
        min_wall_dist = float('inf')
        closest_rect_wall = None
        d_list = []

        for wall in rect_wall_list:
            d = (self.x - wall.center[0])**2 + (self.y - wall.center[1])**2
            d_list.append(d)
            if d < min_wall_dist:
                min_wall_dist = d
                closest_rect_wall = wall
        if closest_rect_wall == None:
            print("get wall state d list point invoke , edge case maybe " , d_list)
            print("self x , self y" , self.x , self.y)
            print("rect wall list ", rect_wall_list)
        closest_dist = get_closest_dist_pt_wrt_rect(self.getPos() , closest_rect_wall) -self.radius   
        # print(f"Agent id {self.id} , Closest Wall Distance : {closest_dist}")
        return StaticObstacleState(closest_rect_wall.topleft , closest_rect_wall.topright , closest_rect_wall.bottomleft , closest_rect_wall.bottomright , closest_dist)
    
    def get_static_obs_state(self,rect_obstacle_list):
        min_obs_dist = float('inf')
        closest_rect_obs = None 
        for obs in rect_obstacle_list:
            d = (self.x - obs.center[0])**2 + (self.y - obs.center[1])**2
            if d < min_obs_dist:
                min_obs_dist = d
                closest_rect_obs = obs 

        closest_dist = get_closest_dist_pt_wrt_rect(self.getPos(),closest_rect_obs) -self.radius
        # print(f"Agent id {self.id}, Closest Ostacle Dist : {closest_dist}")
        return StaticObstacleState(closest_rect_obs.topleft , closest_rect_obs.topright , closest_rect_obs.bottomleft , closest_rect_obs.bottomright , closest_dist)
    
    def getGoal(self):
        return self.goal
    
    def assign_next_subgoal(self , agent_list): 
        # print("len self.sgl" , len(self.sgl))
        # print("id : ," ,self.id)
        # print("curr subgoal idx : ",self.curr_subgoal_idx)    
        # print("subgoal list : ",self.sgl)  
        # print("goal :" , self.goal.getPos())
        # print("self is final goal reached :" , self.is_final_goal_reached)
        if  self.curr_subgoal_idx < len(self.sgl)-1 :
            g = self.sgl[self.curr_subgoal_idx+1]
            self.curr_subgoal_idx +=1
            self.curr_subgoal = g

        elif self.curr_subgoal_idx == len(self.sgl)-1 and self.is_final_goal_reached == False:
            # print("yoooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo")
            self.is_final_goal_reached = True
    
    def chkandupdate_subgoal(self , agent_list):
        max_sg_idx = -1

        for agent in agent_list:
            if agent.id != self.id:
                if agent.grp_id == self.grp_id:
                    if agent.curr_subgoal_idx > max_sg_idx:
                        max_sg_idx = agent.curr_subgoal_idx
        if self.id == 0:
            print("max sg idx ",max_sg_idx)
        if max_sg_idx > self.curr_subgoal_idx:
            self.curr_subgoal_idx = max_sg_idx
            self.curr_subgoal = self.sgl[self.curr_subgoal_idx]

    def determine_entering_exiting(self , graph_point_inside_office_dict , graph_point_infront_office_dict):
        subgoal_list_len = len(self.sgl)
        case = None
        office_idx = -1
        idx1 = 0 # for exiting
        idx2 = subgoal_list_len - 3 # for entering
        
        if self.curr_subgoal_idx == idx1+1:
            case="exiting"
            a = graph_point_inside_office_dict[self.sgl[idx1]]
            b = graph_point_infront_office_dict[self.sgl[idx1+1]]

            if a != -1 and b !=-1:
                if a == b:
                    office_idx = a-1
                    return case , office_idx
    
        if self.curr_subgoal_idx == idx2 + 1:
            case = "entering"
                    
            a = graph_point_infront_office_dict[self.sgl[idx2]]
            b = graph_point_inside_office_dict[self.sgl[idx2+1]]

            if a != -1 and b !=-1:
                if a == b:
                    office_idx = a-1
                    return case , office_idx
    
       
        return case ,office_idx

    def is_safe_to_enter(self , agent_list  , gate, graph_point_inside_office_dict , graph_point_infront_office_dict):
        if gate.gate_office_idx == 0 or gate.gate_office_idx == 5:
            inner_y = 75
            outer_y = 30
            
            tl = (gate.x1 , gate.y1 - inner_y)
            tr = (gate.x2 , gate.y2 - inner_y)
            bl = (gate.x1 , gate.y1 + outer_y)
            br = (gate.x2 , gate.y2 + outer_y)

        else:
            inner_y = 75
            outer_y = 30 
            tl = (gate.x1 , gate.y1 - outer_y)
            tr = (gate.x2 , gate.y2 - outer_y)
            bl = (gate.x1 , gate.y1 + inner_y)
            br = (gate.x2 , gate.y2 + inner_y)
        # box_dimensions
        
        for agent in agent_list:   
            if agent != self:
               case , oid2 = agent.determine_entering_exiting(graph_point_inside_office_dict , graph_point_infront_office_dict)
               if check_pt_in_box(tl , tr , bl , br , agent.getPos()) and case == "exiting":
                    return False
        return True

    def draw_arrow(self,  WINDOW , force_angle , particular_pos = None):
        if particular_pos!=None:
            x , y = particular_pos[0] , particular_pos[1]
        else:
            x , y = self.x , self.y
        r = 13.5
        if force_angle == None:
            return
        pt1_x = self.radius * np.cos(force_angle) + x
        pt1_y = self.radius * np.sin(force_angle) + y
       
        pt2_x = r * np.cos(force_angle)  + x
        pt2_y = r * np.sin(force_angle) + y
        # print("force angle" , force_angle)
        # print("point 1 " , pt1_x , pt1_y)
        # print("point 2  " ,pt2_x , pt2_y)

        pygame.draw.line(WINDOW , (255,0,0) , (pt1_x,pt1_y),(pt2_x , pt2_y) , width = 2)

    def draw_agent(self,WINDOW , font,particular_pos = None):
        if particular_pos != None:
            pygame.draw.circle(WINDOW , self.color ,(particular_pos[0],particular_pos[1]),self.radius)    
        else:
            pygame.draw.circle(WINDOW , self.color ,(self.x,self.y),self.radius)
        text = font.render(f"{self.id}", True, (0, 0, 0))
        if particular_pos!=None:
            WINDOW.blit(text , text.get_rect(center = (particular_pos[0],particular_pos[1])))
        else:
            WINDOW.blit(text , text.get_rect(center = (self.x,self.y)))
    
       
    def assign_path_to_agent(self , graph_points , graph_dict):
        sgl = []
        agent_office_idx = chk_agent_in_which_office(office_list , self.x , self.y)
        goal_office_idx = chk_agent_in_which_office(office_list , self.goal.x , self.goal.y)


        if agent_office_idx == goal_office_idx:
            sgl.append(self.goal.getCenter())
            self.sgl = sgl 
           
        else:
            if agent_office_idx == -1:
                spt = get_closest_graph_point_wrt_pos((self.x  ,self.y) , (self.goal.x , self.goal.y) , graph_points ,office_list)
            else:
                spt = closest_graph_point_wrt_office[agent_office_idx]
            # print("SPt " , spt)
            if goal_office_idx == -1:
                ept = get_closest_graph_point_wrt_pos((self.goal.x , self.goal.y) ,(self.x , self.y)  , graph_points , office_list)
            else:
                ept = closest_graph_point_wrt_office[goal_office_idx]
            # print("EPF " , ept)
            sgl = get_path_astar(spt , ept , graph_dict)
            sgl.append(self.goal.getCenter())
            
            self.sgl = sgl

    def compute_gate_walls(self , gate_list):
        # this function to be run only at starting at agent initialization
        goal_office_idx = chk_agent_in_which_office(office_list , self.goal.x , self.goal.y)-1
        agent_office_idx = chk_agent_in_which_office(office_list , self.x , self.y)-1
        gates_to_convert = []
        for gate in gate_list:
            if gate.gate_office_idx != goal_office_idx and gate.gate_office_idx != agent_office_idx:
                gates_to_convert.append(gate)
        gate_walls = []
        for g in gates_to_convert:
            gate_walls.append(g.translate_to_wall())
        self.gate_walls = gate_walls

    def update(self):
        if self.is_final_goal_reached:
            return
          
        self.x += (self.vx)
        self.y += (self.vy)
        self.action = ActionXY(self.vx , self.vy)
        # if self.id == 0:
        #     print(self.x , self.y)
        #     print("self. prev ", self.prev) 
        if len(self.prev) == 2:
            if abs(round(self.x,2) - round(self.prev[0][0],2)) < 0.005 and abs(round(self.y,2) - round(self.prev[0][1],2)) < 0.005:
                print("Random Force")
                # print("self x , y agent on which random ",self.x,self.y)
                apply_random_force(self)
                # print("after applying random -- > ",self.x , self.y)
                # print("self. prev  for agent on which random acted ", self.prev)
        if len(self.prev) == 2:      
            self.prev.pop(0)
        self.prev.append((self.x,self.y))
        

        if self.x < 0:
            self.x = 0
        if self.x > WIDTH:
            self.x = WIDTH - self.radius-3
        if self.y < 0:
            self.y = 0
        if self.y > HEIGHT:
            self.y = HEIGHT - self.radius-3
        
    def is_step_safe(self , agent_list , rect_obstacle_list , rect_wall_list):
        virt_x = self.virt_x
        virt_y = self.virt_y
        for agent in agent_list:
            if self != agent:
                d = np.sqrt((virt_x - agent.x)**2 + (virt_y - agent.y)**2)
                if d < 2 * self.radius:
                    # print(f"Step not safe btw :agent {agent.id} , {self.id}")
                    return False
        
        # for r in rect_obstacle_list:
        #     if chk_circle_rect_collision((virt_x,virt_y),self.radius ,r):
        #         print("#################################### obs")
        #         return False 

        # for w in rect_wall_list:
        #     if chk_circle_rect_collision((virt_x,virt_y),self.radius,w):
        #         print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ wall")
        #         return False
        return True

    # move agent using action
    def move_agent_using_action(self, action , delta = 1):
        self.vx =  action.vx
        self.vy =  action.vy
        self.virt_x = self.x + self.vx * delta
        self.virt_y = self.y + self.vy * delta
        angle = math.atan2(self.vy, self.vx)
        return angle

    def predict_action_linear_policy(self , state):
        self_state = state.self_state
        theta = np.arctan2(self_state.gy-self_state.py, self_state.gx-self_state.px)
        vx = np.cos(theta) * self_state.v_pref
        vy = np.sin(theta) * self_state.v_pref
        action = ActionXY(vx, vy)

        return action

    # move agent using spf
    def move_agent(self,agent_list,wall_list , bc_list , rect_obstacle_list):
        # print(f"Agent Idx : {self.id} , Agent Subgoal Idx : {self.curr_subgoal_idx}")
        if self.isGoalReached() == True and self.waiting_mode == False:
            self.assign_next_subgoal(agent_list)
            # print("hello bro")
            return [],[],None
        
        # self.chkandupdate_subgoal(agent_list)
        F_attr , fl_attr = compute_F_attr_total(self,agent_list , bc_list)
        F_rep  , fl_rep = compute_F_rep_total(self , agent_list ,rect_obstacle_list)
        
        # worthy_walls = find_worthy_walls(wall_list , self.x , self.y)
        # f_wall_x  ,f_wall_y = 0 , 0
        # for item in worthy_walls:
        #     f_rep_x , f_rep_y = find_rep_force_from_wall(item[0] , self.x , self.y ,item[1])
        #     f_wall_x += f_rep_x
        #     f_wall_y += f_rep_y
        f_wall_x , f_wall_y = find_rep_force_from_walls_v2(self , wall_list , self.x , self.y , self.radius)
        fl_rep.append(("wall rep",f_wall_x, f_wall_y , math.atan2(f_wall_y,  f_wall_x)))
        f_left_x , f_left_y = leftward_force_on_agent(bc_list , self)
        # print(f"left force {f_left_x , f_left_y}")
        # # print("wall rep , ",f_wall_x , f_wall_y)
        F_rep_tx = F_rep[0] + f_wall_x
        F_rep_ty = F_rep[1] + f_wall_y 
        F_total = (F_attr[0] + F_rep_tx + f_left_x,F_attr[1] + F_rep_ty + f_left_y)
        
        ## get state 
        F_total_magnitude = compute_magnitude(F_total)
        if F_total_magnitude == 0:
            F_total_magnitude = 0.0001
        force_vector = (F_total[0]/F_total_magnitude , F_total[1]/F_total_magnitude)
        # print("force vector " , force_vector )
        force_angle = math.atan2(force_vector[1] , force_vector[0])
        V = 1
        vdx = round((F_total[0] / F_total_magnitude) * V,2)
        vdy = round((F_total[1]/ F_total_magnitude) * V,2)

        
        self.vx =  vdx 
        self.vy =  vdy
        self.virt_x = self.x + self.vx 
        self.virt_y = self.y + self.vy
        
        # self.x += vdx
        # self.y += vdy
        # if len(self.prev) == 2:
        #     if abs(round(self.x,2) - round(self.prev[0][0],2)) < 0.005 and abs(round(self.y,2) - round(self.prev[0][1],2)) < 0.005:
        #         print("Random Force")
        #         apply_random_force(self)
        
        # if len(self.prev) == 2:      
        #     self.prev.pop(0)
        # self.prev.append((self.x,self.y))
        

        # if self.x < 0:
        #     self.x = 0
        # if self.x > WIDTH:
        #     self.x = WIDTH - self.radius-2
        # if self.y < 0:
        #     self.y = 0
        # if self.y > HEIGHT:
        #     self.y = HEIGHT - self.radius-2
        
        return fl_attr , fl_rep , force_angle
