from scipy.spatial import ConvexHull
from state import *
import numpy as np

# for one group and 1 agent check
def violation_check(state):
    
    self_state = state.self_state
    other_states = state.other_states
    grp_agent_cords = []
    lone_agent_pos = None
    if self_state.grp_id == 0:  
        grp_agent_cords.append(self_state.position)
        for state in other_states:
            if state.grp_id == 0:
                grp_agent_cords.append(state.position)
            if state.grp_id == 1:
                lone_agent_pos = state.position
    elif self_state.grp_id == 1:
        lone_agent_pos = self_state.position
        for state in other_states:
            if state.grp_id == 0:
                grp_agent_cords.append(state.position)
    
    # compute convex hull of group
    grp_agent_cords_np = np.array(grp_agent_cords)
    if len(grp_agent_cords) <= 2:
        return False
    convex_hull = ConvexHull(grp_agent_cords_np)
    convex_hull_verts = grp_agent_cords_np[convex_hull.vertices]
    # print("vertices dekh lo")
    # print(convex_hull_verts)
    # print("grp_agent_cords")
    # print(grp_agent_cords)
    hull_points = []
    for i in range(convex_hull_verts.shape[0]):
        hull_points.append((convex_hull_verts[i,0],convex_hull_verts[i,1]))
    if pip_check(hull_points , lone_agent_pos):
        return True
    return False


# for 2 groups of multiple size check 
# def violation_check(state):
    
#     self_state = state.self_state
#     other_states = state.other_states

#     grp_agent_cords1 = []
#     grp_agent_cords2 = []
#     grp_agent_cords1.append(self_state.position)
#     for state in other_states:
#         if state.grp_id == 0:
#             grp_agent_cords1.append(state.position)
#         if state.grp_id == 1:
#             grp_agent_cords2.append(state.position) 

#     # print(grp_agent_cords1)
#     # print(grp_agent_cords2)              
#     # compute convex hull of group
#     grp_agent_cords_np1 = np.array(grp_agent_cords1)
#     if len(grp_agent_cords1) > 1:
#         convex_hull1 = ConvexHull(grp_agent_cords_np1)
#         convex_hull_verts1 = grp_agent_cords_np1[convex_hull1.vertices]
    
#     grp_agent_cords_np2 = np.array(grp_agent_cords2)
#     if len(grp_agent_cords2) > 1:
#         convex_hull2 = ConvexHull(grp_agent_cords_np2)
#         convex_hull_verts2 = grp_agent_cords_np2[convex_hull2.vertices]
   
#     # print("vertices dekh lo")
#     # print(convex_hull_verts)
#     # print("grp_agent_cords")
#     # print(grp_agent_cords)
#     if len(grp_agent_cords1) > 1:
#         hull_points1 = []
#         for i in range(convex_hull_verts1.shape[0]):
#             hull_points1.append((convex_hull_verts1[i,0],convex_hull_verts1[i,1]))

#     if len(grp_agent_cords2) > 1:
#         hull_points2 = []
#         for i in range(convex_hull_verts2.shape[0]):
#             hull_points2.append((convex_hull_verts2[i,0],convex_hull_verts2[i,1]))

#     if len(grp_agent_cords1) == 1 and len(grp_agent_cords2) == 1:
#         return False


#     if len(grp_agent_cords2) > 1:   
#         print("helllooooooooooo")
#         for pos in grp_agent_cords1:
#             if pip_check(hull_points2 , pos):
#                 return True
#     if len(grp_agent_cords1) > 1:
#         print("geloooooooooooooo")
#         for pos in grp_agent_cords2:
#             if pip_check(hull_points1 , pos):
#                 return True

#     return False

def pip_check(hull_points , lone_pt):
    coef_list = []
    for i in range(len(hull_points)):
        p1_x , p1_y = hull_points[i][0] , hull_points[i][1]
        p2_x , p2_y = hull_points[(i+1)%len(hull_points)][0] , hull_points[(i+1)%len(hull_points)][1]

        # line eq : ax + by + c = 0
        a = p2_y - p1_y
        b = p1_x - p2_x
        c = p1_y * p2_x - p1_x * p2_y
        coef_list.append((a,b,c))

    point_in_line_vals = []
    for i in range(len(coef_list)):
         line_params = coef_list[i]
         point_in_line_val = line_params[0] * lone_pt[0] + line_params[1] * lone_pt[1] + line_params[2]
         point_in_line_vals.append(point_in_line_val)
    
    flag1 = all( val >= 0 for val in point_in_line_vals) 
    flag2 = all(val <= 0 for val in point_in_line_vals)
    
    if flag1 or flag2:
        return True
    return False