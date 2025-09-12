import numpy as np
import time
import pathlib
HOME_PATH = str(pathlib.Path("../").parent.resolve())
import sys
sys.path.append(HOME_PATH)
from Mujoco_env.envs.h1_base import H1_inspireMed
from assets.robots.h1_inspire import H1_inspire
import copy as cp
import utils_.KDL_utils.transform as T

class H1_inspireMed_Pos_Ctrl(H1_inspireMed):
    """

    :param is_render: Choose if use the renderer to render the scene or not.
    :param renderer: Choose official renderer with "viewer",
            another renderer with "mujoco_viewer"
    :param control_freq: Upper-layer control frequency.
            Note that high frequency will cause high time-lag.
    :param cam_view: Choice of Camera view.
    """

    def __init__(self,
                 robot=None,
                 is_render=False,
                 renderer="viewer",
                 control_freq=200,
                 is_interpolate=True,
                 cam_view=None
                 ):
        super().__init__(
            robot=robot,
            is_render=is_render,
            renderer=renderer,
            control_freq=control_freq,
            is_interpolate=is_interpolate,
            cam_view=cam_view
        )


        #计算手掌对于手腕的偏移
        left_eef_pos=self.getBodyPos("left_wrist_yaw_link")
        right_eef_pos=self.getBodyPos("right_wrist_yaw_link")
        left_red_point_pos=self.getBodyPos("L_eef")
        right_red_point_pos=self.getBodyPos("R_eef")

        left_eef_mat = self.getBodyMat("left_wrist_yaw_link")
        right_eef_mat = self.getBodyMat("right_wrist_yaw_link")
        left_red_point_mat=self.getBodyMat("L_eef")
        right_red_point_mat=self.getBodyMat("R_eef")

        left_eef_T=np.eye(4)
        left_eef_T[:3, :3]=left_eef_mat
        left_eef_T[:3, 3]=left_eef_pos

        right_eef_T=np.eye(4)
        right_eef_T[:3, :3]=right_eef_mat
        right_eef_T[:3, 3]=right_eef_pos

        left_red_point_T=np.eye(4)
        left_red_point_T[:3, :3]=left_red_point_mat
        left_red_point_T[:3, 3]=left_red_point_pos

        right_red_point_T=np.eye(4)
        right_red_point_T[:3, :3]=right_red_point_mat
        right_red_point_T[:3, 3]=right_red_point_pos
        

        # 计算左红点相对于左 EEF 的变换
        self.left_bias_T = np.linalg.inv(left_eef_T) @ left_red_point_T

        # 计算右红点相对于右 EEF 的变换
        self.right_bias_T = np.linalg.inv(right_eef_T) @ right_red_point_T

        #vr相关的标志位
        self.collect_start=False
        self.start_flag = False
        self.tele_first = True


        self.cam_start()
        
    def step_jnt(self,action,simple_action=np.zeros(7+7+1+1+1)):
        super().step(action,simple_action)

    def ik_func(self,action):
        l_target=action[0:6]
        r_target=action[6:12]
        action_l=self.ik_solve(l_target,self.arm_left,"left")
        action_r=self.ik_solve(r_target,self.arm_right,"right")
        res=np.hstack((action_l,action_r,action[12:]))
        # print(res)
        return res

    def step_all_simple(self,action_jnt):

        simple_action=action_jnt.copy()
        action_l=action_jnt[0:7]
        action_r=action_jnt[7:14]
        action_troso=action_jnt[14]
        action_l_hand=action_jnt[15]
        action_r_hand=action_jnt[16]


        hand_map_list=[1.3,0.2,0.8,1.2,1.7,1.7,1.7,1.7,1.7,1.7,1.7,1.7]#0~1映射到手指

        action_l_hand1=np.array([1.3, 0, 0, 0])
        action_l_hand2=np.array([0.0,0.0])
        action_l_hand3=np.array([0.0,0.0])
        action_l_hand4=np.array([0.0,0.0])
        action_l_hand5=np.array([0.0,0.0])
        action_r_hand1=np.array([1.3, 0, 0, 0])
        action_r_hand2=np.array([0.0,0.0])
        action_r_hand2=np.array([0.0,0.0])
        action_r_hand3=np.array([0.0,0.0])
        action_r_hand4=np.array([0.0,0.0])
        action_r_hand5=np.array([0.0,0.0])
        #手指控制
        for i in range(4):
            if(i!=0):#不控制拇指第一个关节
                action_l_hand1[i]=action_l_hand * hand_map_list[i]
                action_r_hand1[i]=action_r_hand * hand_map_list[i]
        for i in range(2):
            action_l_hand2[i]=action_l_hand * hand_map_list[4+i]
            action_l_hand3[i]=action_l_hand * hand_map_list[6+i]
            action_l_hand4[i]=action_l_hand * hand_map_list[8+i]
            action_l_hand5[i]=action_l_hand * hand_map_list[10+i]
            action_r_hand2[i]=action_r_hand * hand_map_list[4+i]
            action_r_hand3[i]=action_r_hand * hand_map_list[6+i]
            action_r_hand4[i]=action_r_hand * hand_map_list[8+i]
            action_r_hand5[i]=action_r_hand * hand_map_list[10+i]
        
        action = np.hstack((action_l,action_r,
                            action_troso,
                            action_l_hand1,action_l_hand2,action_l_hand3,action_l_hand4,action_l_hand5,
                            action_r_hand1,action_r_hand2,action_r_hand3,action_r_hand4,action_r_hand5))
        
        self.step_all(action,simple_action=simple_action)

        
    def reset(self,box_pos=[0.4,0.05,1.03]):
        
        self.mj_data.joint('red_box_joint').qpos[0] = box_pos[0] 
        self.mj_data.joint('red_box_joint').qpos[1] = box_pos[1]
        self.mj_data.joint('red_box_joint').qpos[2] = box_pos[2]
        self.mj_data.joint('red_box_joint').qpos[3] = 1.0
        self.mj_data.joint('red_box_joint').qpos[4] = 0.0
        self.mj_data.joint('red_box_joint').qpos[5] = 0.0
        self.mj_data.joint('red_box_joint').qpos[6] = 0.0
        super().reset()
        self.cam_flage = True
        t=0
        while self.cam_flage:
            if(type(self.top)==type(None) 
               or type(self.angle)==type(None) ):
                time.sleep(0.001)
                t+=1
            else:
                self.cam_flage=False

    def update_pc(self):
        left_pc = np.zeros(7)
        right_pc = np.zeros(7)
        for i in range(7):
            left_pc[i] = self.mj_data.joint(self.arm_left.joint_index[i]).qpos[0]
            right_pc[i] = self.mj_data.joint(self.arm_right.joint_index[i]).qpos[0]
        return left_pc,right_pc

    def preStep(self, action):
        if isinstance(action,np.ndarray) and (len(action)==14 or len(action)>14):
            super().preStep(action)
        else:
            raise AttributeError("size should be 14")

    def get_box_state(self):
        box_pose = np.zeros(3)
        for i in range(3):
            box_pose[i] = cp.deepcopy(self.mj_data.joint('red_box_joint').qpos[i])
        return box_pose
        
    def ik_solve(self,target,Arm,arm_type):
        pos_goal=target[0:3]
        rot_goal=target[3:]
        pos_bias,mat_bias = Arm.get_Arm_bias_frame()
        pos_bias = np.array([pos_bias[0],pos_bias[1],pos_bias[2]])
        mat_bias = np.array([[mat_bias[0],mat_bias[1],mat_bias[2]],
                            [mat_bias[3],mat_bias[4],mat_bias[5]],
                            [mat_bias[6],mat_bias[7],mat_bias[8]]])
        mat_bias = np.linalg.inv(mat_bias)
        mat_goal = T.vec2Mat(rot_goal)
        p_goal = mat_bias@(pos_goal-pos_bias)
        mat_goal = mat_bias@mat_goal

        #是否eef偏移到红点，即让红点到达target的位置(红点位置已隐藏，位于手掌中心)
        #######################################
        if(arm_type=="left"):
            ee_bias_T=self.left_bias_T.copy()
            
        elif(arm_type=="right"):
            ee_bias_T=self.right_bias_T.copy()
            # print(ee_bias_T)
        
        goal_T=np.eye(4)
        goal_T[:3, :3]=mat_goal.copy()
        goal_T[:3, 3]=p_goal.copy()

        goal_T=goal_T.copy()@np.linalg.inv(ee_bias_T)

        p_goal=goal_T[:3, 3].copy()
        mat_goal=goal_T[:3, :3].copy()
        #######################################

        return Arm.kdl_solver.ikSolver(p_goal, mat_goal, Arm.arm_qpos)    
    
    def step_all(self,action,simple_action=np.zeros(7+7+1+1+1)):
        self.step_jnt(action,simple_action)  
        if self.is_render:
            self.render()

def make_sim_env(freq=60,_is_render=True):
    env = H1_inspireMed_Pos_Ctrl(
        robot=H1_inspire(),
        is_render=_is_render,
        control_freq=freq,
        is_interpolate=True,
        cam_view='double_cam'
    )

    return env




if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)

    env = make_sim_env()
    env.reset()

    action_l = np.zeros(7)
    action_r = np.zeros(7)
    
    left_target=np.array([0.244,  0.21353,  1.14161,   #左臂目标xyz、rpy
                          0.0, 0.0, 0.0])
    right_target=np.array([0.244,  -0.21353,  1.14161,   #左臂目标xyz、rpy  
                          0.0, 0.0, 0.0])
    action_troso=np.ones(1)*0.0 #腰转动角度

    action_l_hand1=np.array([1.3, 0, 0, 0])
    action_l_hand2=np.array([0,0])
    action_l_hand3=np.array([0,0])
    action_l_hand4=np.array([0,0])
    action_l_hand5=np.array([0,0])
    action_r_hand1=np.array([1.3, 0, 0, 0])
    action_r_hand2=np.array([0,0])
    action_r_hand2=np.array([0,0])
    action_r_hand3=np.array([0,0])
    action_r_hand4=np.array([0,0])
    action_r_hand5=np.array([0,0])

    action = np.hstack((action_l,action_r,
                        action_troso,
                        action_l_hand1,action_l_hand2,action_l_hand3,action_l_hand4,action_l_hand5,
                        action_r_hand1,action_r_hand2,action_r_hand3,action_r_hand4,action_r_hand5)) 
    action_init=action.copy()

    while (True):

        action_l=env.ik_solve(left_target,env.arm_left,"left")
        action_r=env.ik_solve(right_target,env.arm_right,"right")

        action = np.hstack((action_l,action_r,
                            action_troso,
                            action_l_hand1,action_l_hand2,action_l_hand3,action_l_hand4,action_l_hand5,
                            action_r_hand1,action_r_hand2,action_r_hand3,action_r_hand4,action_r_hand5))  

        #下发关节角度
        env.step_all(action)
            