import math
from queue import PriorityQueue
import numpy as np

class AStar_Node:
    def __init__(self, pos ,val):
        self.parent = None
        self.h = 0
        self.g = 0
        self.f = 0
        self.pos = pos
        self.val  = val
        
    
    def gen_children(self, graph_node_dict):
        children = []
        # print("inside gen child ",self.pos)
        children = graph_node_dict.get(self , None)
       
        # print(len(children))
        # for child in children:
        #     print(child.pos)
        return children

    # def gen_children(self, nn_grid):
    #     rows = len(nn_grid)
    #     cols = len(nn_grid[0])
    #     children = []
    #     for dir in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
    #         new_pos = (self.pos[0] + dir[0] , self.pos[1] + dir[1])
    #         if new_pos[0] > (rows-1) or new_pos[0] < 0 or new_pos[1] > (cols-1) or new_pos[1] < 0:
    #             continue
    #         children.append(nn_grid[new_pos[0]][new_pos[1]])

    #     return children

    def __lt__(self, other):
        if self.f < other.f:
            return True
        else:
            return False


def H(pt1 ,pt2):
    if pt1 == (624,327) and pt2 == (115,385):
        return 264330
    # print("hooristic : ,",pt1 , pt2)
    h = math.sqrt(abs(pt1[0]-pt2[0])**2 + abs(pt1[1]-pt2[1])**2)
    # print("h is : ",h)
    
    return h


# def make_node_grid(cpt_grid):
#     grid = cpt_grid.tolist()
#     for i in range(len(grid)):
#         for j in range(len(grid[i])):
#             grid[i][j] = AStar_Node((i,j),grid[i][j])
#     return grid

def print_qlist(l):
    for item in l:
        print(f"[Priority:{item[0]},tie:{item[1]},pos:{item[2].pos}]",end=" ")
    print()

def astar2(start_node, goal_node, cpt_grid):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    
    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        print(current_node.pos)
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)
        
        # Found the goal
        if current_node == goal_node:
            path =  []
            while current_node.parent:
                path.append(current_node.pos)
                current_node = current_node.parent
            path.append(current_node.pos)
            return path[::-1]

        # Generate children
        children = current_node.gen_children(cpt_grid)
        for c in children:
            print(f"Children : {c.pos} , parent :  {c.parent}")
        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = H(child.pos , goal_node.pos)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)

def astar(start_node , goal_node , cpt_grid):
    open_list = PriorityQueue()
    closed_list = set()
    open_set = set()
    tie_breaker = 0
    open_list.put((0 ,tie_breaker , start_node))
    
    open_set.add(start_node)
    
    while (open_list.empty() == False):
        # temp_list = list(open_list.queue)
        # print_qlist(temp_list)
        curr_node = open_list.get()[2]
        
        # print(curr_node.f)
        open_set.remove(curr_node)

        if curr_node == goal_node:
            path =  []
            while curr_node.parent:
                path.append(curr_node.pos)
                curr_node = curr_node.parent
            path.append(curr_node.pos)
            return path[::-1]
            
        closed_list.add(curr_node)
        # if curr_node.pos == (629,518):
            # print("BHAISAHAB ################")
        children = curr_node.gen_children(cpt_grid)
        flag = False
        if children == None:
            continue
        for child in children:
            # if curr_node.pos == (736,515):
            #         print("yo ",child.pos)
            # if curr_node.pos == (624,327):
            #         print("bo ",child.pos)
            
            if child in closed_list:
                continue
            if child in open_set:
                temp = curr_node.g + 1
                if child.pos == (629,518):
                    print("629 , 518 g2 val :" ,temp)
                if temp < child.g:
                    child.g = temp
                    child.parent = curr_node
                    child.h = H(child.pos , goal_node.pos)
                    child.f = child.g + child.h
            else:    
                child.g = curr_node.g + 1
                # if child.pos == (624,327):
                #     print("624,327 g val : ",child.g)
                # if child.pos == (629,518):
                #     print("629 , 518 g val :" ,child.g)
                
               
                child.parent = curr_node
                child.h = H(child.pos , goal_node.pos)
                # if child.pos == (624,327):
                #     print("624,327 h val : ",child.h)
                # if child.pos == (629,518):
                #     print("629 , 518 h val :" ,child.h)
                
                # print(goal_node.pos)
                # print(f"curr node pos : {child.pos} , curr node h : {child.h}")
                child.f = child.g + child.h
                tie_breaker += 1
                open_list.put((child.f , tie_breaker , child))
                open_set.add(child)


#   THIS PART IS FOR TESTING THIS MODULE , IF NO USE THEN IGNORE IT
# def main():
#     maze = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])    
    
#     start = (0, 0)
#     end = (1, 1)
#     maze2 = make_node_grid(maze)
#     sn = maze2[start[0]][start[1]]
#     en = maze2[end[0]][end[1]]
#     for i in  range(len(maze2)):
#         for j in range(len(maze2[0])):
#             print(maze2[i][j].pos , end = " ")
#         print()
#     path = []
#     path = astar(sn, en , maze2)

#     for i in range(len(path)):
#         print(path[i].pos)
  
# main()

