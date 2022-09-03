class FullState:
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref , grp_id):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.gx = gx
        self.gy = gy
        self.v_pref = v_pref
        # self.theta = theta

        self.position = (self.px, self.py)
        self.goal_position = (self.gx, self.gy)
        self.velocity = (self.vx, self.vy)
        self.grp_id = grp_id

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref,self.grp_id)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy,
                                          self.v_pref , self.grp_id]])


class ObservableState:
    def __init__(self, px, py, vx, vy, radius , grp_id):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.grp_id = grp_id

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius , self.grp_id)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius,self.grp_id]])
# convecrt to agent centric

# code wall state and static state
# return observation in that format
# convert the rotate function 

class StaticObstacleState:
    def __init__(self,tl , tr , bl , br , closest_dist):
        self.tl = tl 
        self.tr = tr
        self.bl = bl 
        self.br = br 
        self.d_from_agent = closest_dist

    def __str__(self):
        return ' '.join([str(x) for x in [self.tl, self.tr, self.bl, self.br, self.d_from_agent]])
    
    def __add__(self, other):
        return other + (self.tl[0] , self.tl[1] , self.tr[0] , self.tr[1] , self.bl[0],self.bl[1] , self.br[0],self.br[1] ,self.d_from_agent) 

class JointState:
    def __init__(self, self_state, other_states , wall_state = None , static_obs_state = None):
        assert isinstance(self_state, FullState)
        
        for other_state in other_states:
            assert isinstance(other_state, ObservableState)

        self.self_state = self_state
        self.other_states = other_states
        self.wall_state = wall_state
        self.static_obstacle_state = static_obs_state 
