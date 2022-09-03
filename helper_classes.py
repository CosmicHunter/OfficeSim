import numpy as np

# helper classes like wall , office , gate , box cells

class Wall:
    def __init__(self ,id, x1 , y1 , x2 , y2):
        self.id = id
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        
    def print(self):
        return f"Wall {self.x1} {self.y1} {self.x2} {self.y2}"
    
    def get_len(self):
        return np.sqrt((self.x1 - self.x2)**2 + (self.y1-self.y2)**2)
    
    def is_fake_wall(self):
        if self.id == -1:
            return True
        return False

    def get_nearest_pt(self , px , py):
        rel_wall_vec_x = self.x2 - self.x1
        rel_wall_vec_y = self.y2 - self.y1

        rel_pos_vec_x = px - self.x1
        rel_pos_vec_y = py - self.y1
        rel_wall_vec_len = np.sqrt(rel_wall_vec_x**2 + rel_wall_vec_y**2)
        unit_rel_wall_vec = (rel_wall_vec_x / rel_wall_vec_len , rel_wall_vec_y / rel_wall_vec_len)

        rel_pos_scale_vec = (rel_pos_vec_x * (1 /rel_wall_vec_len) , rel_pos_vec_y * (1/rel_wall_vec_len))

        dotprod = unit_rel_wall_vec[0] * rel_pos_scale_vec[0] + unit_rel_wall_vec[1] * rel_pos_scale_vec[1]

        if dotprod < 0:
            nearest_pt = (self.x1 , self.y1)
        elif dotprod > 1:
            nearest_pt = (self.x2 , self.y2)
        else:
            nearest_pt = (rel_wall_vec_x * dotprod + self.x1 , rel_wall_vec_y * dotprod + self.y1)
        
        return nearest_pt



class Office:
    def __init__(self ,id, bl , br , tl , tr):
        self.id = id
        self.bl = bl
        self.br = br
        self.tl = tl
        self.tr = tr


class Gate:
    def __init__(self,id, x1  , y1 , x2 , y2 , goid):
        self.id = id
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.gate_office_idx = goid

    def translate_to_wall(self):
        return Wall(-1 , self.x1  , self.y1 , self.x2 , self.y2)
       
# boxcell1 (45,478) (945,478), (45,561) ,(945,561)
# boxcell2  (754,44) (945,44)  (754,171) (945,171)
class BoxCell:
    def __init__(self , id  , tl , tr , bl , br):
        self.id = id
        self.tl = tl
        self.tr = tr
        self.bl = bl
        self.br = br

