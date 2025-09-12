import numpy as np
import matplotlib.pyplot as plt



class PolicyBase:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None
        # print("init base policy")

    def generate_trajectory(self, obj_state):   
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = np.array(curr_waypoint['xyz'])
        curr_euler = curr_waypoint['euler']
        curr_troso = curr_waypoint['troso']
        curr_hand = curr_waypoint['hand']
        next_xyz = np.array(next_waypoint['xyz'])
        next_euler = next_waypoint['euler']
        next_troso = next_waypoint['troso']
        next_hand = next_waypoint['hand']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        euler = curr_euler + (next_euler - curr_euler) * t_frac
        troso=curr_troso + (next_troso - curr_troso) * t_frac
        hand = curr_hand + (next_hand - curr_hand) * t_frac
        return xyz, euler,troso, hand

    def predict(self, obj_state, step_count):

        # generate trajectory at first timestep, then open-loop execution
        self.step_count = step_count
        if self.step_count == 0:
            self.generate_trajectory(obj_state)
            
            action_l=np.array([0.244,  0.21353,  1.14161,   #左臂目标xyz、rpy
                                0.0, 0.0, 0.0])
            action_r=np.array([0.244,  -0.21353,  1.14161,   #左臂目标xyz、rpy  
                                0.0, 0.0, 0.0])    
            
            action_hand = np.array([0.0,0.0])
            action_troso = np.array([0.0])
            action = np.hstack((action_l,action_r,action_troso,action_hand))
            return action

        elif self.step_count >= 1: 
            # print(self.left_trajectory[0]['t'])
            if self.left_trajectory[0]['t'] == self.step_count:
                self.curr_left_waypoint = self.left_trajectory.pop(0)
            next_left_waypoint = self.left_trajectory[0]

            if self.right_trajectory[0]['t'] == self.step_count:
                self.curr_right_waypoint = self.right_trajectory.pop(0)
            next_right_waypoint = self.right_trajectory[0]

            # Inject noise
            if self.inject_noise:
                scale = 0.01
                left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
                right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

            # interpolate between waypoints to obtain current pose and hand command
            left_xyz, left_euler, troso, left_hand = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)
            right_xyz, right_euler, troso, right_hand = self.interpolate(self.curr_right_waypoint, next_right_waypoint, self.step_count)
            action_hand = np.array([left_hand,right_hand])
            action_left = np.concatenate([left_xyz, left_euler])
            action_right = np.concatenate([right_xyz, right_euler])
            self.step_count += 1
            # print(np.hstack([action_left, action_right, troso, action_hand]))
            return np.hstack([action_left, action_right, troso, action_hand])#troso只在right中定义有效

        
        else:
            raise AttributeError
        

class H1Policy(PolicyBase):
    def __init__(self, inject_noise=False,task_name=""):
        super().__init__(inject_noise)
        self.task_name=task_name
    def generate_trajectory(self, obj_state):


        if(self.task_name == "Transmitting the red stick"):

            l_start_xyz=np.array([0.344,  0.21353,  1.14161])
            l_start_euler = np.array([0.0,0.0,0.0])
            r_start_xyz=np.array([0.344,  -0.21353,  1.14161])
            r_start_euler = np.array([0.0,0.0,0.0])

            l_ready_xyz = np.array([0.374,  0.25353,  1.24161])
            l_ready_euler = np.array([0.0,0.0,0.0])
            r_ready_xyz = np.array([0.374,  -0.21353,  1.14161])
            r_ready_euler = np.array([0.0,0.0,0.0])

            l_meetready_xyz = np.array([0.394,  0,  1.2561])
            l_meetready_euler = np.array([0.0,0.0,0.0])
            r_meetready_xyz = np.array([0.374,  -0.05,  1.12161])
            r_meetready_euler = np.array([0.0,0.0,0.0])

            peg_info = obj_state[:6]
            peg_xyz = peg_info[:3] #物体位置

            self.left_trajectory = [
                {"t": 1, "xyz": l_start_xyz, "euler":l_start_euler, "troso": 0, "hand": 0}, 
                {"t": 100, "xyz": l_ready_xyz, "euler":l_start_euler, "troso": 0, "hand": 0}, 
                {"t": 200, "xyz": peg_xyz+np.array([0,0,0.035]), "euler":l_start_euler, "troso": 0, "hand": 0},
                {"t": 230, "xyz": peg_xyz+np.array([0,0,0.035]), "euler":l_start_euler, "troso": 0, "hand": 1},
                {"t": 250, "xyz": peg_xyz+np.array([0,0,0.035]), "euler":l_start_euler, "troso": 0, "hand": 1},
                {"t": 300, "xyz": peg_xyz+np.array([0,0,0.2]), "euler":l_start_euler, "troso": 0, "hand": 1}, 
                {"t": 400, "xyz": l_meetready_xyz, "euler":l_start_euler, "troso": 0, "hand": 1}, 
                {"t": 430, "xyz": l_meetready_xyz+np.array([0,0,-0.08]), "euler":l_start_euler, "troso": 0, "hand": 1}, 
                {"t": 440, "xyz": l_meetready_xyz+np.array([0,0,-0.08]), "euler":l_start_euler, "troso": 0, "hand": 1}, 
                {"t": 460, "xyz": l_meetready_xyz+np.array([0,0,-0.08]), "euler":l_start_euler, "troso": 0, "hand": 1}, 
                {"t": 500, "xyz": l_meetready_xyz+np.array([0,0,-0.08]), "euler":l_start_euler, "troso": 0, "hand": 0}, 
                {"t": 550, "xyz": l_start_xyz+np.array([-0.1,0,0]), "euler":l_start_euler, "troso": 0, "hand": 0},
                {"t": 600, "xyz": l_start_xyz, "euler":l_start_euler, "troso": 0, "hand": 0}, 
            ]

            self.right_trajectory = [
                {"t": 1, "xyz": r_start_xyz, "euler": r_start_euler, "troso": 0, "hand": 0}, 
                {"t": 100, "xyz": r_ready_xyz, "euler": r_start_euler, "troso": 0, "hand": 0},
                {"t": 200, "xyz": r_ready_xyz, "euler": r_start_euler, "troso": 0, "hand": 0},
                {"t": 300, "xyz": r_ready_xyz, "euler": r_start_euler, "troso": 0, "hand": 0},
                {"t": 400, "xyz": r_meetready_xyz, "euler": r_start_euler, "troso": 0, "hand": 0},
                {"t": 440, "xyz": r_meetready_xyz, "euler": r_start_euler, "troso": 0, "hand": 0},
                {"t": 460, "xyz": r_meetready_xyz, "euler": r_start_euler, "troso": 0, "hand": 1},
                {"t": 470, "xyz": r_meetready_xyz, "euler": r_start_euler, "troso": 0, "hand": 1},
                {"t": 500, "xyz": r_meetready_xyz, "euler": r_start_euler, "troso": 0, "hand": 1},
                {"t": 550, "xyz": r_start_xyz, "euler": r_start_euler, "troso": 0, "hand": 1},
                {"t": 600, "xyz": r_start_xyz, "euler": r_start_euler, "troso": 0, "hand": 1},

            ]






            

        #demo1
        # if(self.task_name == "Clean table"):
        #     l_start_xyz=np.array([0.344,  0.21353,  1.14161])
        #     l_start_euler = np.array([0.0,0.0,0.0])
        #     r_start_xyz=np.array([0.344,  -0.21353,  1.14161])
        #     r_start_euler = np.array([0.0,0.0,0.0])

        #     l_ready_xyz = np.array([0.374,  0.25353,  1.24161])
        #     l_ready_euler = np.array([0.0,0.0,0.0])
        #     r_ready_xyz = np.array([0.374,  -0.21353,  1.14161])
        #     r_ready_euler = np.array([0.0,0.0,0.0])

        #     l_meetready_xyz = np.array([0.7,  0.15,  1.2561])
        #     l_meetready_euler = np.array([0.0,0.0,0.0])
        #     r_meetready_xyz = np.array([0.374,  -0.05,  1.12161])
        #     r_meetready_euler = np.array([0.0,0.0,0.0])

        #     peg_info = obj_state[:6]
        #     peg_xyz = peg_info[:3] #物体位置

        #     self.left_trajectory = [
        #         {"t": 1, "xyz": l_start_xyz, "euler":l_start_euler, "troso": 0, "hand": 0}, 
        #         {"t": 100, "xyz": l_ready_xyz, "euler":l_start_euler, "troso": 0, "hand": 0}, 
        #         {"t": 200, "xyz": peg_xyz+np.array([0,0,0.035]), "euler":l_start_euler, "troso": 0, "hand": 0},
        #         {"t": 230, "xyz": peg_xyz+np.array([0,0,0.035]), "euler":l_start_euler, "troso": 0, "hand": 1},
        #         {"t": 250, "xyz": peg_xyz+np.array([0,0,0.035]), "euler":l_start_euler, "troso": 0, "hand": 1},
        #         {"t": 300, "xyz": peg_xyz+np.array([0,0,0.2]), "euler":l_start_euler, "troso": 0, "hand": 1}, 
        #         {"t": 400, "xyz": l_meetready_xyz, "euler":l_start_euler, "troso": 0, "hand": 1}, 
        #         {"t": 430, "xyz": l_meetready_xyz+np.array([0,0,-0.08]), "euler":l_start_euler, "troso": 0, "hand": 1}, 
        #         {"t": 440, "xyz": l_meetready_xyz+np.array([0,0,-0.08]), "euler":l_start_euler, "troso": 0, "hand": 1}, 
        #         {"t": 460, "xyz": l_meetready_xyz+np.array([0,0,-0.08]), "euler":l_start_euler, "troso": 0, "hand": 0}, 
        #         {"t": 500, "xyz": l_meetready_xyz+np.array([0,0,-0.08]), "euler":l_start_euler, "troso": 0, "hand": 0}, 
        #         {"t": 550, "xyz": l_start_xyz+np.array([-0.1,0,0]), "euler":l_start_euler, "troso": 0, "hand": 0},
        #         {"t": 580, "xyz": l_start_xyz, "euler":l_start_euler, "troso": 0, "hand": 0}, 
        #     ]

        #     self.right_trajectory = [
        #         {"t": 1, "xyz": r_start_xyz, "euler": r_start_euler, "troso": 0, "hand": 0}, 
        #         {"t": 100, "xyz": r_start_xyz, "euler": r_start_euler, "troso": 0, "hand": 0},
        #         {"t": 200, "xyz": r_start_xyz, "euler": r_start_euler, "troso": 0, "hand": 0},
        #         {"t": 300, "xyz": r_start_xyz, "euler": r_start_euler, "troso": 0, "hand": 0},
        #         {"t": 400, "xyz": r_start_xyz, "euler": r_start_euler, "troso": -0.4, "hand": 0},
        #         {"t": 430, "xyz": r_start_xyz, "euler": r_start_euler, "troso": -0.4, "hand": 0},
        #         {"t": 450, "xyz": r_start_xyz, "euler": r_start_euler, "troso": -0.4, "hand": 0},
        #         {"t": 460, "xyz": r_start_xyz, "euler": r_start_euler, "troso": -0.4, "hand": 0},
        #         {"t": 500, "xyz": r_start_xyz, "euler": r_start_euler, "troso": -0.4, "hand": 0},
        #         {"t": 550, "xyz": r_start_xyz, "euler": r_start_euler, "troso": 0, "hand": 0},
        #         {"t": 580, "xyz": r_start_xyz, "euler": r_start_euler, "troso": 0, "hand": 0},

        #     ]
        #demo2
        if(self.task_name == "Clean table"):   
            l_start_xyz=np.array([0.344,  0.21353,  1.14161])
            l_start_euler = np.array([0.0,0.0,0.0])
            r_start_xyz=np.array([0.344,  -0.21353,  1.14161])
            r_start_euler = np.array([0.0,0.0,0.0])

            l_ready_xyz = np.array([0.374,  0.25353,  1.24161])
            l_ready_euler = np.array([0.0,0.0,0.0])
            r_ready_xyz = np.array([0.374,  -0.21353,  1.14161])
            r_ready_euler = np.array([0.0,0.0,0.0])

            l_meetready_xyz = np.array([0.7,  0.15,  1.2561])
            l_meetready_euler = np.array([0.0,0.0,0.0])
            r_meetready_xyz = np.array([0.374,  -0.05,  1.12161])
            r_meetready_euler = np.array([0.0,0.0,0.0])

            peg_info = obj_state[:6]
            peg_xyz = peg_info[:3] #物体位置

            self.left_trajectory = [
                {"t": 1, "xyz": l_start_xyz, "euler":l_start_euler, "troso": 0, "hand": 0}, 
                {"t": 50, "xyz": l_ready_xyz, "euler":l_start_euler, "troso": 0, "hand": 0}, 
                {"t": 150, "xyz": peg_xyz+np.array([0,0,0.035]), "euler":l_start_euler, "troso": 0, "hand": 0},
                {"t": 230, "xyz": peg_xyz+np.array([0,0,0.035]), "euler":l_start_euler, "troso": 0, "hand": 1},
                {"t": 250, "xyz": peg_xyz+np.array([0,0,0.035]), "euler":l_start_euler, "troso": 0, "hand": 1},
                {"t": 350, "xyz": peg_xyz+np.array([0,0,0.2]), "euler":l_start_euler, "troso": 0, "hand": 1}, 
                {"t": 400, "xyz": l_meetready_xyz, "euler":l_start_euler, "troso": 0, "hand": 1}, 
                {"t": 430, "xyz": l_meetready_xyz+np.array([0,0,-0.08]), "euler":l_start_euler, "troso": 0, "hand": 1}, 
                {"t": 440, "xyz": l_meetready_xyz+np.array([0,0,-0.08]), "euler":l_start_euler, "troso": 0, "hand": 1}, 
                {"t": 460, "xyz": l_meetready_xyz+np.array([0,0,-0.08]), "euler":l_start_euler, "troso": 0, "hand": 0}, 
                {"t": 500, "xyz": l_meetready_xyz+np.array([0,0,-0.08]), "euler":l_start_euler, "troso": 0, "hand": 0}, 
                {"t": 520, "xyz": l_start_xyz+np.array([-0.1,0,0]), "euler":l_start_euler, "troso": 0, "hand": 0},
                {"t": 580, "xyz": l_start_xyz, "euler":l_start_euler, "troso": 0, "hand": 0}, 
            ]

            self.right_trajectory = [
                {"t": 1, "xyz": r_start_xyz, "euler": r_start_euler, "troso": 0, "hand": 0}, 
                {"t": 100, "xyz": r_start_xyz, "euler": r_start_euler, "troso": 0, "hand": 0},
                {"t": 200, "xyz": r_start_xyz, "euler": r_start_euler, "troso": 0, "hand": 0},
                {"t": 300, "xyz": r_start_xyz, "euler": r_start_euler, "troso": 0, "hand": 0},
                {"t": 400, "xyz": r_start_xyz, "euler": r_start_euler, "troso": -0.4, "hand": 0},
                {"t": 430, "xyz": r_start_xyz, "euler": r_start_euler, "troso": -0.4, "hand": 0},
                {"t": 450, "xyz": r_start_xyz, "euler": r_start_euler, "troso": -0.4, "hand": 0},
                {"t": 460, "xyz": r_start_xyz, "euler": r_start_euler, "troso": -0.4, "hand": 0},
                {"t": 500, "xyz": r_start_xyz, "euler": r_start_euler, "troso": -0.4, "hand": 0},
                {"t": 550, "xyz": r_start_xyz, "euler": r_start_euler, "troso": 0, "hand": 0},
                {"t": 580, "xyz": r_start_xyz, "euler": r_start_euler, "troso": 0, "hand": 0},

            ]

