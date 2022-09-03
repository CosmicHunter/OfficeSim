import pygame
from helper_for_testsim import *
from pipcheck_modified import *


def check_pt_in_boxcell(bc , pos):
    if pos[0] >= bc.tl[0] and pos[0] <= bc.tr[0]:
        if pos[1] >= bc.tl[1] and pos[1] <= bc.bl[1]:
            return True
    return False

def check_pt_in_box(tl , tr , bl ,br , pos):
    # print(pos)
    if pos[0] >=tl[0] and pos[0] <= tr[0]:
        if pos[1] >= tl[1] and pos[1] <= bl[1]:
            return True
    return False


def check_if_pos_safe(pos , agent_list , rect_obstacle_list):
    flag = True
    if len(agent_list) !=0:
        for agent in agent_list:
            if compute_dist(pos , agent.getPos()) < 2 * agent.radius + 5:
                return False
    radius = 9
    for r in rect_obstacle_list:
            if chk_circle_rect_collision(pos ,radius ,r):
                return False    
    return True

# Checks the position of agent in a group is safe from rect obstacle and rect walls
def check_if_pos_safe_v2(pos , agent_list , rect_obstacle_list, rect_wall_list):
    flag = True
   
    radius = agent_list[0].radius
    for r in rect_obstacle_list:
        if chk_circle_rect_collision(pos ,radius ,r):
            # print("####################################")
            return False 

    for w in rect_wall_list:
        if chk_circle_rect_collision(pos , radius , w):
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            return False


    return True

def chk_circle_rect_collision(pos ,radius, rect):
    x_dist = abs(pos[0] - rect.center[0])
    y_dist = abs(pos[1] - rect.center[1])
    
    if x_dist > (rect.w/2 + radius):
        return False
    if y_dist > (rect.h/2 + radius):
        return False
    
    if x_dist <= (rect.w/2):
        return True
    if y_dist <= (rect.h/2):
        return True
    
    temp = (x_dist - rect.w/2)**2 + (y_dist - rect.h/2)**2
    return temp <= (radius**2)

def chk_agent_in_office_map(px  , py):
    if px >= office_left_x and px <= office_right_x:
        if py >= office_bottom_y and py <= office_top_y1:
            return True
        
        if py >= office_top_y1 and py <= office_top_y2:
            if px >= office_right_x1:
                return True

    return False

def chk_agent_in_which_office(office_list , px , py):
    for idx in range(len(office_list)):
        off_obj = office_list[idx]
        if off_obj.tl[0] <= px and px <= off_obj.tr[0]:
            if off_obj.tl[1] <= py and py <= off_obj.bl[1]:
                return (idx+1)
    return -1    


def chk_collision_with_rect_obs(rect_obstacle_list , pt):
    for ob in rect_obstacle_list:
        if ob.collidepoint(pt):
            return True
    return False

def generate_goal_pos(office_list , rect_obstacle_list):
    numoffices = len(office_list)
    
    rand_office_idx = np.random.randint(0 , numoffices)
    office_obj = office_list[rand_office_idx]
    
    gx = np.random.randint(office_obj.tl[0]+4 , office_obj.tr[0]-4)
    gy = np.random.randint(office_obj.tl[1]+4 , office_obj.bl[1]-4)
    
    while chk_collision_with_rect_obs(rect_obstacle_list , (gx , gy)) == True:
        rand_office_idx = np.random.randint(0 , numoffices)
        office_obj = office_list[rand_office_idx]
    
        gx = np.random.randint(office_obj.tl[0]+4 , office_obj.tr[0]-4)
        gy = np.random.randint(office_obj.tl[1]+4 , office_obj.bl[1]-4)

    return gx , gy

def leftward_force_on_agent(bc_list , agent):
    if check_pt_in_boxcell(bc_list[0] , agent.getPos()) == False:
        return 0 , 0
    if agent.curr_subgoal_idx + 1 > len(agent.sgl)-1:
        return 0 , 0 
    pos_in_corridor = None
    boxcell = bc_list[0]
    mid_y = (boxcell.tl[1] + boxcell.bl[1])/2
    corridor_width = abs(boxcell.tl[1] - boxcell.bl[1])
    if agent.y + agent.radius <= mid_y:
        pos_in_corridor = "upper"
    else:
        pos_in_corridor = "lower"
    
    force_to_be_applied = False
    # based on left to right or right to left force is applied
    x1 = agent.sgl[agent.curr_subgoal_idx][0]
    x2 = agent.sgl[agent.curr_subgoal_idx+1][0]
    if x1 < x2:
        direction = "L2R"
    elif x1 > x2:
        direction = "R2L"
    else:
        direction = "verti"
    
    if direction == "L2R" and pos_in_corridor == "lower":
        #  print("Bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbboooooooooooooooooooooooooooooo")
         dir_v_y = (boxcell.tl[1] - boxcell.bl[1]) / abs(boxcell.tl[1] - boxcell.bl[1])
         force_to_be_applied = True
    elif direction == "R2L" and pos_in_corridor == "upper":
         # force to be applied downwards
        #  print("gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg")
         dir_v_y =(boxcell.bl[1] - boxcell.tl[1]) / abs(boxcell.bl[1] - boxcell.tl[1])
         force_to_be_applied = True

    if force_to_be_applied:
        force_x = 0
        a = 3
        b = 0.1
        # print(math.exp(-corridor_widt / b))
        # print(f"c /b : {corridor_width/b}")
        magnitude = 1000#a * math.exp(-0.2/ b)
    
        # print(f"MAAAAAAAAAAAAAAAAAAAAG {magnitude}")
        force_y = magnitude  * dir_v_y   
        # print(f"dirvy :{dir_v_y}")     
    else:
        force_x = 0
        force_y = 0

    return force_x , force_y    

    
def find_rep_force_from_walls_v2(agent, wall_list , px , py,radius):
    a = 3
    b = 0.1
    closest_wall_dist_sq = sys.maxsize
    closest_wall_vec = None
    for wall in wall_list+agent.gate_walls:
        npt = wall.get_nearest_pt(px,py)
        vec_npt_to_pt_x = px - npt[0]
        vec_npt_to_pt_y = py - npt[1]
        
        sq_dist = vec_npt_to_pt_x **2 + vec_npt_to_pt_y **2
        if sq_dist < closest_wall_dist_sq:
            closest_wall_dist_sq = sq_dist
            closest_wall_vec = (vec_npt_to_pt_x,vec_npt_to_pt_y)

    min_wall_dist = np.sqrt(closest_wall_dist_sq) - radius
    # Formula = f = a * exp(-min_wall_dist / b)
    force = a * math.exp(-min_wall_dist/b)
    len_closest_wall_vec = np.sqrt(closest_wall_vec[0] **2 + closest_wall_vec[1]**2)
    unit_closest_wall_vec = (closest_wall_vec[0]/len_closest_wall_vec , closest_wall_vec[1]/len_closest_wall_vec)

    force_x = force * unit_closest_wall_vec[0]
    force_y = force * unit_closest_wall_vec[1]
    return force_x,force_y


def generate_rect_obstacles(obstacle_list):
    rect_obstacle_list = []
    for item in obstacle_list:
        left = item[0][0]
        top = item[0][1]
        w = abs(item[0][0] - item[2][0])
        h = abs(item[0][1] - item[1][1])
        rect_obj = pygame.Rect((left , top) , (w , h))
        rect_obstacle_list.append(rect_obj)
    return rect_obstacle_list


def generate_rect_walls(wall_list):
    rect_wall_list = []
    wall_rect_width = 4
    for wall in wall_list:
        if wall.x1 == wall.x2:
            left = wall.x1 - wall_rect_width
            top = wall.y1
            w = 2 * wall_rect_width
            h = abs(wall.y1 - wall.y2)


        elif wall.y1 == wall.y2:
            left = wall.x1
            top = wall.y1 - wall_rect_width
            w = abs(wall.x1 - wall.x2)
            h = 2 * wall_rect_width

        rect_obj = pygame.Rect((left , top) , (w , h))
        rect_wall_list.append(rect_obj)

    return rect_wall_list


def get_closest_graph_point_wrt_pos(pos , goal_pos , graph_points , office_list):
    d = sys.maxsize

    cpt = None
    scpt = None
    for pt in graph_points:
        if chk_agent_in_which_office(office_list , pt[0],pt[1]) != -1:
            continue
        temp = np.sqrt((pos[0] - pt[0])**2 + (pos[1]-pt[1])**2)
        if temp < d:
            d = temp
            scpt = cpt
            cpt = pt 
    
    d1 = np.sqrt((cpt[0]-goal_pos[0])**2 + (cpt[1]-goal_pos[1])**2)
    d2 = sys.maxsize
    if scpt!=None:
        d2 = np.sqrt((scpt[0]-goal_pos[0])**2 + (scpt[1]-goal_pos[1])**2)
    
    if d2 < d1:
        cpt = scpt
    return cpt

def get_node_by_cordinate(pos , node_list):
        for node in node_list:
            if pos == node.pos:
                return node
        return None


def generate_grp_positions(offset , grp_size):
    circle_radius = 20
    init_theta = 0
    theta = init_theta
    delta_theta = 2 * np.pi / grp_size
    offset1 = offset[0]
    offset2 = offset[1]
    grp_positions = []
    for i in range(grp_size):
        x = circle_radius * math.cos(theta) + offset1
        y = circle_radius * math.sin(theta) + offset2
        theta += delta_theta
        grp_positions.append((x,y))
    return grp_positions

def chk_group_positions_safe(grp_positions , agent_list , rect_obstacle_list):
    for grp_pos in grp_positions:
        xpos,ypos = grp_pos[0] , grp_pos[1]
        if check_if_pos_safe((xpos,ypos) , agent_list , rect_obstacle_list) == False:
            return False
    return True


# change scales
def spf_reward(state):
    # can be used when group 1 has only one agent
    self_state = state.self_state
    # if self_state.grp_id == 1:
    #     return 0 , 0
    # print("self state grp id ",self_state.grp_id)
    other_agent_state = state.other_states
    centroid_x  = self_state.px
    centroid_y = self_state.py
    # convention is last human state will be of lonely robot
    centroid_other_grp_x , centroid_other_grp_y = 0  , 0
    grp_size1 , grp_size2 = 1 , 0
    for agent in other_agent_state[:]:
        if agent.grp_id == self_state.grp_id:
           centroid_x += agent.px
           centroid_y += agent.py
           grp_size1+=1
        else:
           centroid_other_grp_x += agent.px 
           centroid_other_grp_y += agent.py
           grp_size2+=1

    centroid_x /= grp_size1
    centroid_y /= grp_size1
    centroid_other_grp_x /= grp_size2
    centroid_other_grp_y /= grp_size2
    reward = 0
    min_safe_dist =  20 #1.1  
    penalty_factor = 0.3
    incentive_for_spf = 0.003
    threshold_barrier = 9.5 #0.35
    dist = np.sqrt((self_state.px - centroid_x)**2 + (self_state.py - centroid_y)**2)   ## add sqrt
    absdiff = abs(dist - min_safe_dist) - self_state.radius
    flag = 0
    if grp_size1 == 1:
        # print(f"for agent " , self_state.grp_id)
        return 0 , 0


    if (dist > min_safe_dist and absdiff > threshold_barrier):
        reward = -(absdiff - threshold_barrier) * penalty_factor / 10
        flag = -1
  
    if (dist < min_safe_dist and absdiff > threshold_barrier):
       reward = -(absdiff - threshold_barrier) * penalty_factor / 10
       flag = -1
   
    elif (dist == min_safe_dist or absdiff <= threshold_barrier):
        reward = incentive_for_spf
        flag = 1
    return reward,flag

def avoid_group_break_reward(state):
    # This function is written with an assumption that we pass the state of the robot that is in a different group (i.e group id = 1)
    lonely_robot_state = state.self_state
    centroid_x  = 0
    centroid_y = 0
    group_agent_states = state.other_states
    # Trigger if the group com is within a certain distance
    other_grp_size = 0
    other_group_agent_states = []
    for agent_state in group_agent_states:
       if lonely_robot_state.grp_id != agent_state.grp_id:
            centroid_x += agent_state.px
            centroid_y += agent_state.py
            other_grp_size+=1
            other_group_agent_states.append(agent_state)
    centroid_x /= other_grp_size
    centroid_y /= other_grp_size
    if other_grp_size == 1:
        return 0
    dist_rob_centroid = np.sqrt((lonely_robot_state.px - centroid_x)**2 + (lonely_robot_state.py - centroid_y)**2)
    # print("----------->",dist_rob_centroid)
    if dist_rob_centroid < 50:
        dist_list = []
        max_dist = 0
        for agent_state in other_group_agent_states:
            temp = np.sqrt((agent_state.px - centroid_x)**2 + (agent_state.py - centroid_y)**2)
            if temp > max_dist:
                max_dist = temp
            dist_list.append(temp)

        val = (lonely_robot_state.px - centroid_x)**2 + (lonely_robot_state.py - centroid_y)**2
        # print("--------------->" ,val <= max_dist**2)
        if val <= (max_dist **2) and violation_check(state) == True:
            
            return -0.18
        else:
            return 0.004
    else:
        return 0 


def get_closest_dist_pt_wrt_rect(point , rect):
    xdist , ydist = 0 , 0
    if point[0] < rect.left:
        xdist = rect.left - point[0]
    elif point[0] > rect.right:
        xdist = point[0] - rect.right
    else:
        xdist = 0

    if point[1] < rect.top:
        ydist = rect.top - point[1]
    elif point[1] > rect.bottom:
        ydist = point[1] - rect.bottom
    else:
        ydist = 0

    if xdist == 0:
        return ydist 
    elif ydist == 0:
        return xdist

    return np.sqrt(xdist**2 + ydist**2)


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0