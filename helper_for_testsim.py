from ssl import OP_NO_COMPRESSION
import pygame
import math
import random
import sys
import time 
import numpy as np
WIDTH = 992
HEIGHT = 892

# this module contains the social force implementation.
# here the force functions are used for social potential field algorithm

# Hyperparameters
K_attr_intergroup = 0
K_attr_intragroup = 200 # 180
K_rep_intergroup =  200
K_rep_intragroup =  190
sigma_attr_intergroup = 3
sigma_attr_intragroup = 0.8
sigma_rep_intergroup = 1
sigma_rep_intragroup = 1


dsoc_intergroup =  38
dsoc_intragroup =  25
# add force log , to build graph and free body diag
# add arrows for force

class Goal:
     def __init__(self, x , y , color):
         self.x = x
         self.y = y
         self.w = 10
         self.h = 10

         self.color_idx = -1
        #  self.color = (0,250,0)
         self.color = color
         self.isAgent = False
         self.radius = self.w/2
     
     def getPos(self):
         return (self.x,self.y)
     
     def getCenter(self):
         return (self.x + self.w/2 , self.y + self.h/2)

     def draw_goal(self,WINDOW , particular_pos = None):
        if particular_pos != None:
            pygame.draw.rect(WINDOW , self.color ,pygame.Rect(particular_pos[0],particular_pos[1],self.w,self.h),0)
        else:
            pygame.draw.rect(WINDOW , self.color ,pygame.Rect(self.x,self.y,self.w,self.h),0)


def apply_random_force(agent):
    dx = round(random.uniform(0.5,1.2),2)
    dy = round(random.uniform(0.5,1.2),2)
    
    agent.x += dx
    agent.y += dy


def compute_dist(pt1 , pt2):
    return math.sqrt(((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2))

def chk_pt_in_boxcell(bc , pos):
    if pos[0] >= bc.tl[0] and pos[1] <= bc.tr[0]:
        if pos[1] >= bc.tl[1] and pos[1] <= bc.bl[1]:
            return True
    return False
def compute_F_attractive_goal_v2(agent, goal , sgl_flag , bclist):
    mode_flag = "normal"
    d_threshold = 50
    K_attr = 25
    if sgl_flag == False:
        dist = compute_dist(agent.getPos(),goal.getCenter())
        goal_pos = goal.getCenter()
    elif sgl_flag:
        dist = compute_dist(agent.getPos(),goal)
        goal_pos = goal
    
    F_attr_x = 0
    F_attr_y = 0
    agent_pos = agent.getPos()
    for bc in bclist:
        if chk_pt_in_boxcell(bc , agent_pos) and chk_pt_in_boxcell(bc , goal_pos):
            mode_flag = "boxcell"
            break
    if mode_flag == "boxcell" and sgl_flag == True:
        gx = goal_pos[0]
        gy = agent_pos[1]
        # print("boxcell goal mode , goal changed !!")
        goal_pos = (gx , gy)
    # goal_pos = goal.getCenter()
    if( dist < d_threshold):
        F_attr_x = -K_attr * (agent_pos[0]-goal_pos[0])
        F_attr_y = -K_attr * (agent_pos[1]-goal_pos[1])
    elif(dist >= d_threshold):
        F_attr_x = -d_threshold * K_attr * ((agent_pos[0]-goal_pos[0]) / dist)
        F_attr_y = -d_threshold * K_attr * ((agent_pos[1]-goal_pos[1]) / dist)
    
    F_attr_x = int(F_attr_x)
    F_attr_y = int(F_attr_y)
    
    return (F_attr_x,F_attr_y)


def compute_F_attractive_goal(agent, goal , sgl_flag):
    d_threshold = 50
    K_attr = 25
    if sgl_flag == False:
        dist = compute_dist(agent.getPos(),goal.getCenter())
        goal_pos = goal.getCenter()
    elif sgl_flag:
        dist = compute_dist(agent.getPos(),goal)
        goal_pos = goal
    F_attr_x = 0
    F_attr_y = 0
    agent_pos = agent.getPos()
    # goal_pos = goal.getCenter()
    if( dist < d_threshold):
        F_attr_x = -K_attr * (agent_pos[0]-goal_pos[0])
        F_attr_y = -K_attr * (agent_pos[1]-goal_pos[1])
    elif(dist >= d_threshold):
        F_attr_x = -d_threshold * K_attr * ((agent_pos[0]-goal_pos[0]) / dist)
        F_attr_y = -d_threshold * K_attr * ((agent_pos[1]-goal_pos[1]) / dist)
    
    F_attr_x = int(F_attr_x)
    F_attr_y = int(F_attr_y)
    
    return (F_attr_x,F_attr_y)

# Task is to add a feeble left ward force to the agents
# that would prevent blockage
#
def compute_F_attractive(agent1, agent2):
    K_attr = 0
    dsoc = 0
    sigma = 0
    if agent1.grp_id == agent2.grp_id:
        K_attr = K_attr_intragroup
        dsoc = dsoc_intragroup
        sigma = sigma_attr_intragroup
    else:
        K_attr = K_attr_intergroup
        dsoc = dsoc_intergroup
        sigma = sigma_attr_intergroup
    
    dist = compute_dist(agent1.getPos(),agent2.getPos())
    F_attr_x = 0
    F_attr_y = 0
    if dist > dsoc:
        agent1_pos = agent1.getPos()
        agent2_pos = agent2.getPos()
        
        F_attr_x = -K_attr * (1 / dist ** sigma)  * (agent1_pos[0] - agent2_pos[0])
        F_attr_y = -K_attr * (1 / dist ** sigma) * (agent1_pos[1] - agent2_pos[1])
    
    F_attr_x = int(F_attr_x)
    F_attr_y = int(F_attr_y)
    
    return (F_attr_x,F_attr_y)


def orientation_pt_wrt_rect_obs(obstacle , pt):
    x = pt[0] 
    y = pt[1]
    if x < obstacle.left or x > obstacle.right:
        if y < obstacle.top or y > obstacle.bottom:
            return "diagonal"
        if y >= obstacle.top and y<= obstacle.bottom:
            return "hori"
    elif x>= obstacle.left and x <= obstacle.right:
        if y < obstacle.top or y > obstacle.bottom:
            return "verti"
    
    return "inside"


def get_closest_rect_corner(x , y , rect):
    min_dist = sys.maxsize
    closest_corner = None

    for corner in [rect.topleft , rect.bottomleft , rect.topright , rect.bottomright]:
        d = (x - corner[0])**2 + (y - corner[1])**2
        if d < min_dist:
            min_dist = d
            closest_corner = corner
    return closest_corner

def get_closest_rect_pt(ori ,pos , obstacle):
    
    if ori == "hori":
        if pos[0] >= obstacle.right:
            closest_pt = (obstacle.right , pos[1])
        elif pos[0] <= obstacle.left:
            closest_pt = (obstacle.left , pos[1])
    elif ori == "verti":
        if pos[1] <= obstacle.top:
            closest_pt = (pos[0] , obstacle.top)
        elif pos[1] >= obstacle.bottom:
            closest_pt = (pos[0] , obstacle.bottom)
    # print(f"closest point to obs with center {obstacle.center} is {closest_pt}")
    return closest_pt

# def compute_F_repulsive_obs(agent , obstacle):
#     # d_rep  = 500
#     d_rep = 200
#     K_rep = 5000000
#     F_rep_x = 0.0
#     F_rep_y = 0.0
#     ori = orientation_pt_wrt_rect_obs(obstacle , (agent.x , agent.y))
    
#     dist_subtracter = 0
#     dist = 0
#     if ori == "hori":
#         dist_subtracter = obstacle.width / 2
#         obstacle_pos = get_closest_rect_pt(ori , (agent.x , agent.y) , obstacle)
#         dist = compute_dist(agent.getPos(),obstacle_pos) - agent.radius
#     elif ori == "verti":
#         dist_subtracter = obstacle.height / 2
#         obstacle_pos = get_closest_rect_pt(ori, (agent.x ,agent.y) , obstacle)
#         dist = compute_dist(agent.getPos(),obstacle.center) - agent.radius
#     elif ori == "diagonal":
#         dist = compute_dist(agent.getPos() , get_closest_rect_corner(agent.x , agent.y , obstacle)) - agent.radius
#         obstacle_pos = get_closest_rect_corner(agent.x , agent.y , obstacle)
#     else:
#         print("wrong wrong wrong wrong")
#     # dist = compute_dist(agent.getPos(),obstacle.center)
#     agent_pos = agent.getPos()
#     if dist <= d_rep and dist > 0:
#         # print("hooooooooooooooooooooooooooooooooooooooooooooooooooooooooola")
#         # print("dist ",dist)
#         #  k rep * exp(-d) * vector
#         F_rep_x = K_rep * ( (1.0 / dist) - (1.0 / d_rep)) * ((agent_pos[0] - obstacle_pos[0]) / (dist**3))  
#         F_rep_y = K_rep * ( (1.0 / dist) - (1.0 / d_rep)) * ((agent_pos[1] - obstacle_pos[1]) / (dist**3))
#         # print(F_rep_x , F_rep_y)
#     F_rep_x = round(F_rep_x , 2)
#     F_rep_y = round(F_rep_y , 2)
#     # F_rep_x = int(F_rep_x)
#     # F_rep_y = int(F_rep_y)
#     # print("f repppppppppppppppus ",F_rep_x , F_rep_y)

#     return (F_rep_x,F_rep_y)


def compute_F_repulsive_obs(agent , obstacle):
    # d_rep  = 500
    d_rep = 50
    K_rep = 500
    sigma = 10
    F_rep_x = 0.0
    F_rep_y = 0.0
    ori = orientation_pt_wrt_rect_obs(obstacle , (agent.x , agent.y))
    
    dist_subtracter = 0
    dist = 0
    if ori == "hori":
        dist_subtracter = obstacle.width / 2
        obstacle_pos = get_closest_rect_pt(ori , (agent.x , agent.y) , obstacle)
        dist = compute_dist(agent.getPos(),obstacle_pos) - agent.radius
    elif ori == "verti":
        dist_subtracter = obstacle.height / 2
        obstacle_pos = get_closest_rect_pt(ori, (agent.x ,agent.y) , obstacle)
        dist = compute_dist(agent.getPos(),obstacle_pos) - agent.radius
    elif ori == "diagonal":
        dist = compute_dist(agent.getPos() , get_closest_rect_corner(agent.x , agent.y , obstacle)) - agent.radius
        obstacle_pos = get_closest_rect_corner(agent.x , agent.y , obstacle)
    else:
        obstacle_pos = obstacle.center
        dist = compute_dist(agent.getPos() , obstacle_pos)
        # print("Wong")
    agent_pos = agent.getPos()
    if dist <= d_rep:
        F_rep_x = K_rep * math.exp(-dist / sigma) * ((agent_pos[0] - obstacle_pos[0]) / (dist))  
        F_rep_y = K_rep * math.exp(-dist/ sigma) * ((agent_pos[1] - obstacle_pos[1]) / (dist))
    F_rep_x = round(F_rep_x , 2)
    F_rep_y = round(F_rep_y , 2)
  
    return (F_rep_x,F_rep_y)

def compute_F_repulsive(agent1, agent2):
    K_rep = 0
    dsoc = 0
    sigma = 0

    if agent1.grp_id == agent2.grp_id:
        K_rep = K_rep_intragroup
        dsoc = dsoc_intragroup
        sigma = sigma_rep_intragroup
    else:
        K_rep = K_rep_intergroup
        dsoc = dsoc_intergroup
        sigma = sigma_rep_intergroup
   
    dist = compute_dist(agent1.getPos(),agent2.getPos()) - agent1.radius - agent2.radius
    # if dist < 0:
    #     print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO Bhai")
    #     print(dist)
    #     print(agent1.radius)
    #     print(agent2.radius)
    F_rep_x = 0
    F_rep_y = 0
    if dist < dsoc:
        agent1_pos = agent1.getPos()
        agent2_pos = agent2.getPos()
        
        F_rep_x = K_rep * (1 / dist ** sigma)  * (agent1_pos[0] - agent2_pos[0])
        F_rep_y = K_rep * (1 / dist ** sigma) * (agent1_pos[1] - agent2_pos[1])

    F_rep_x = int(F_rep_x)
    F_rep_y = int(F_rep_y)
    
    return (F_rep_x , F_rep_y)

def compute_F_attr_total(agent , agent_list , bc_list):
    F_attr_total_x = 0
    F_attr_total_y = 0
    force_list_attr = [] # convention is # attraction to agents , attraction to goal
  
    for agent2 in agent_list:
        if (agent is agent2) == False:
            temp = compute_F_attractive(agent , agent2)
            
            F_attr_total_x += temp[0]
            F_attr_total_y += temp[1]
    force_list_attr.append(("agent attr",F_attr_total_x , F_attr_total_y , math.atan2(F_attr_total_y , F_attr_total_x)))
    if agent.waiting_mode:
        temp = compute_F_attractive_goal(agent, agent.getPos() , True)
    if agent.curr_subgoal != agent.goal.getCenter() and agent.curr_subgoal != None:
        temp = compute_F_attractive_goal(agent , agent.curr_subgoal,True)
    
    elif agent.curr_subgoal == agent.goal.getCenter():
        temp = compute_F_attractive_goal(agent , agent.goal,False)
    force_list_attr.append(("goal attr ",temp,math.atan2(temp[1] , temp[0])))  
    F_attr_total_x += temp[0]
    F_attr_total_y += temp[1]

    # print("attr :" , F_attr_total_x,F_attr_total_y)
    return (F_attr_total_x , F_attr_total_y) , force_list_attr


def compute_F_rep_total(agent, agent_list , obstacle_list):
    F_rep_total_x = 0
    F_rep_total_y = 0
    force_list_rep = [] # convention , (rep from agents , rep from obstacles)
    for agent2 in agent_list:
        if (agent is agent2) == False:
            # if agent2.isReached == False:
                temp = compute_F_repulsive(agent , agent2)
                # print("is Complex  ?  : " , np.iscomplex(temp))
                F_rep_total_x += temp[0]
                F_rep_total_y += temp[1]
    force_list_rep.append(("agent rep ",F_rep_total_x , F_rep_total_y , math.atan2(F_rep_total_y,F_rep_total_x)))
    temp2_x  = 0
    temp2_y = 0
  
    for obstacle in obstacle_list:
        temp = compute_F_repulsive_obs(agent , obstacle)
        # print(f"obstacle center :  {obstacle.center} , force : {temp}")
        
        temp2_x += temp[0]
        temp2_y += temp[1]
        F_rep_total_x += temp[0]
        F_rep_total_y += temp[1]
    force_list_rep.append(("obs rep",temp2_x , temp2_y , math.atan2(temp2_y,temp2_x)))
    
    # print("rep: ",F_rep_total_x,F_rep_total_y)
    return (F_rep_total_x , F_rep_total_y) , force_list_rep


def generate_goal(agent_list):
    x = 0
    y = 0
    while True:
        x = random.randint(20,400)
        y = random.randint(30 ,400)
        flag = False

        for agent in agent_list:
            if (abs(agent.x - x) ** 2 + abs(agent.y - y)**2)<800 and (agent.y != y):
                # print("ho")
                flag = True
        if flag == False:
            
            break       
    goal = Goal(x,y)
    for agent in agent_list:
       agent.goal = goal 
    # print("goal" , x,y)

    return goal



def compute_magnitude(v):
  
    return math.sqrt(v[0]**2 + v[1]**2)
