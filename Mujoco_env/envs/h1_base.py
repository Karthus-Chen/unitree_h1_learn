import numpy as np
import time
import copy as cp
import pathlib
HOME_PATH = str(pathlib.Path("../").parent.resolve())
import sys
sys.path.append(HOME_PATH+'/Mujoco_env')
from envs.mujoco_base import MujocoEnv
from utils_.interpolators import OTG
import mujoco as mj
import threading
import collections
import time
from assets.robots.h1_inspire import H1_inspire
import cv2

class H1_inspireMed(MujocoEnv):
    """

    :param is_render: Choose if use the renderer to render the scene or not.
    :param renderer: Choose official renderer with "viewer",
            another renderer with "mujoco_viewer"
    :param control_freq: Upper-layer control frequency.
            Note that high frequency will cause high time-lag.
    :param is_interpolate: Use interpolator while stepping.
    """

    def __init__(self,
                 robot=None,
                 is_render=False,
                 renderer="viewer",
                 control_freq=20,
                 cam_view=None,
                 is_interpolate=False #是否插值
                 ):
        super().__init__(
            robot=robot,
            is_render=is_render,
            renderer=renderer,
            control_freq=control_freq
        )
        self.arm_left  = self.robot.left_arm
        self.arm_right = self.robot.right_arm
        self.troso = self.robot.torso
        self.l_hand1 = self.robot.l_hand1
        self.l_hand2 = self.robot.l_hand2
        self.l_hand3 = self.robot.l_hand3
        self.l_hand4 = self.robot.l_hand4
        self.l_hand5 = self.robot.l_hand5

        self.r_hand1 = self.robot.r_hand1
        self.r_hand2 = self.robot.r_hand2
        self.r_hand3 = self.robot.r_hand3
        self.r_hand4 = self.robot.r_hand4
        self.r_hand5 = self.robot.r_hand5
        self.base_time = self.control_timestep
        
        self.cam = cam_view
        self.interpolator_left = None
        self.interpolator_right = None
        self.compute_qpos = np.zeros(7+7+1+1+1)
        if is_interpolate:
            self._initInterpolator(self.robot.arms)

        #camera view setting
        self.top = None
        self.angle = None
        self.obs = None



    def actuate_J(self, q_target, qdot_target, Arm,tau_env=0):
        """ Compute desired torque with robot dynamics modeling:
            > M(q)qdd + C(q, qd)qd + G(q) + tau_F(qd) + tau_env = tau_ctrl
                Mujoco give qfrc_bias = C(q, qd)qd + G(q) + tau_F(qd)
                we need to compute  M(q)qdd and tau_env
        :param q_target: joint position
        :param qdot_target: joint velocity
        """
        acc_desire = [
            (self.robot.kp[i] * (q_target[i] - Arm.arm_qpos[i]) -
             (self.robot.kd[i] * Arm.arm_qvel[i])) for i in range(Arm.jnt_num)]
        qM = Arm.kdl_solver.getInertiaMat([Arm.arm_qpos[i] for i in range(len(Arm.joint_index))])
        tau_target = np.dot(qM, acc_desire) + np.array(
            [self.mj_data.joint(Arm.joint_index[i]).qfrc_bias[0] for i in range(len(Arm.joint_index))])
        for i in range(len(Arm.joint_index)):
            self.mj_data.actuator(Arm.actuator_index[i]).ctrl = tau_target[i] +  tau_env 
        
        
    def reset(self):
        self._setInitPose()
        mj.mj_step(self.mj_model, self.mj_data)
        if self.interpolator_left is not None and self.interpolator_right is not None:
            self.interpolator_left.setOTGParam(self.robot.arms[0].arm_qpos,
                                        np.zeros(len(self.robot.arms[0].joint_index)))
            self.interpolator_right.setOTGParam(self.robot.arms[1].arm_qpos,
                                        np.zeros(len(self.robot.arms[1].joint_index)))
        
    def set_joint(self):
        self._setJointPose()
        mj.mj_step(self.mj_model, self.mj_data)


    def step(self,action,simple_action=np.zeros(7+7+1+1+1)):
        self.compute_qpos = simple_action  #for observation !
        self.obs = self._get_obs()
        ctrl_cur_time = time.time()
        for i in range(int(self.control_timestep / self.model_timestep)): #控制次数，一次控制多次下发,mujoco默认model_timestep为1/2000，
            if int(self.control_timestep / self.model_timestep) == 0:
                raise ValueError("Control frequency is too low. Checkout you are not in renderer mode."
                                 "Current Model-Timestep:{}".format(self.model_timestep))
            super().step(action)

        self.base_time = time.time() - ctrl_cur_time
            

    def preStep(self, action):

        if len(action) == 14:  #仅手臂控制
            q_target_l = action[:7]
            q_target_r= action[7:] 
            self.actuate_J(q_target_l, np.zeros(7), self.arm_left)
            self.actuate_J(q_target_r, np.zeros(7), self.arm_right)

            self.actuate_J(np.zeros(1), np.zeros(7), self.troso)
            self.actuate_J(np.zeros(4), np.zeros(7), self.l_hand1)
            self.actuate_J(np.zeros(2), np.zeros(7), self.l_hand2)
            self.actuate_J(np.zeros(2), np.zeros(7), self.l_hand3)
            self.actuate_J(np.zeros(2), np.zeros(7), self.l_hand4)
            self.actuate_J(np.zeros(2), np.zeros(7), self.l_hand5)
            self.actuate_J(np.zeros(4), np.zeros(7), self.r_hand1)
            self.actuate_J(np.zeros(2), np.zeros(7), self.r_hand2)
            self.actuate_J(np.zeros(2), np.zeros(7), self.r_hand3)
            self.actuate_J(np.zeros(2), np.zeros(7), self.r_hand4)
            self.actuate_J(np.zeros(2), np.zeros(7), self.r_hand5)

        elif len(action) > 14:  #手臂+腰+手指控制
            q_target_l = action[:7]
            q_target_r= action[7:14] 
            q_target_troso= action[14]
            q_target_l_hand1= action[15:19]
            q_target_l_hand2= action[19:21]
            q_target_l_hand3= action[21:23]
            q_target_l_hand4= action[23:25]
            q_target_l_hand5= action[25:27]


            q_target_r_hand1= action[27:31]
            q_target_r_hand2= action[31:33]
            q_target_r_hand3= action[33:35]
            q_target_r_hand4= action[35:37]
            q_target_r_hand5= action[37:39]

            q_target_troso=np.ones(7)*q_target_troso.copy()


            self.actuate_J(q_target_l, np.zeros(7), self.arm_left)
            self.actuate_J(q_target_r, np.zeros(7), self.arm_right)
            self.actuate_J(q_target_troso, np.zeros(7), self.troso)
            self.actuate_J(q_target_l_hand1, np.zeros(7), self.l_hand1)
            self.actuate_J(q_target_l_hand2, np.zeros(7), self.l_hand2)
            self.actuate_J(q_target_l_hand3, np.zeros(7), self.l_hand3)
            self.actuate_J(q_target_l_hand4, np.zeros(7), self.l_hand4)
            self.actuate_J(q_target_l_hand5, np.zeros(7), self.l_hand5)

            self.actuate_J(q_target_r_hand1, np.zeros(7), self.r_hand1)
            self.actuate_J(q_target_r_hand2, np.zeros(7), self.r_hand2)
            self.actuate_J(q_target_r_hand3, np.zeros(7), self.r_hand3)
            self.actuate_J(q_target_r_hand4, np.zeros(7), self.r_hand4)
            self.actuate_J(q_target_r_hand5, np.zeros(7), self.r_hand5)

        else:
            print(action)
            print(len(action))
            raise AttributeError


    @property
    def get_obs_qpos(self):
        qpos_left = np.zeros(self.arm_left.jnt_num)
        qpos_right = np.zeros(self.arm_right.jnt_num)
        qpos_troso = np.zeros(1)
        qpos_l_hand = np.zeros(1)
        qpos_r_hand = np.zeros(1)
        for i in range(self.arm_left.jnt_num):
            qpos_left[i] = cp.deepcopy(self.mj_data.joint(self.arm_left.joint_index[i]).qpos[0])
        for i in range(self.arm_right.jnt_num):
            qpos_right[i] = cp.deepcopy(self.mj_data.joint(self.arm_right.joint_index[i]).qpos[0])

        qpos_troso = cp.deepcopy(self.mj_data.joint(self.troso.joint_index[0]).qpos[0])
        qpos_l_hand=cp.deepcopy(self.mj_data.joint(self.l_hand3.joint_index[0]).qpos[0]/1.7)#因为是simple控制，所以拿一个手指采样
        qpos_r_hand=cp.deepcopy(self.mj_data.joint(self.r_hand3.joint_index[0]).qpos[0]/1.7)
        return np.hstack([qpos_left,qpos_right,qpos_troso,qpos_l_hand,qpos_r_hand]) #simple obs


    def _get_obs(self):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_obs_qpos
        obs['action'] = self.compute_qpos
        obs['images'] = dict()
        obs['images']['top']   = self.top
        obs['images']['angle'] = self.angle
        return obs

    def _get_image_obs(self):
        obs = collections.OrderedDict()
        obs['images'] = dict()
        obs['images']['top']   = self.top
        obs['images']['angle'] = self.angle
        return obs
    
    def _get_qpos_obs(self):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_obs_qpos
        return obs
    
    def get_box_state(self):
        raise NotImplementedError



    @property
    def cam_view(self):
        
        if self.cam == 'top':
            return self.top
        elif self.cam == 'angle':
            return self.angle
        elif self.cam == 'double_cam':
            return np.hstack((self.top, self.angle))
        else:
            raise AttributeError("please input right name")


    def camera_viewer(self):
        img_renderer = mj.Renderer(self.mj_model,height=480,width=640)
        if(self.is_render):
            # cv2.namedWindow('Cam view',cv2.WINDOW_NORMAL)
            pass
        while not self.exit_flag:
            img_renderer.update_scene(self.mj_data,camera="top")
            self.top = img_renderer.render()
            self.top = self.top[:, :, ::-1]
            img_renderer.update_scene(self.mj_data,camera="angle")
            self.angle = img_renderer.render()
            self.angle = self.angle[:, :, ::-1]
            
            if(self.is_render):
                # cv2.imshow('Cam view', self.cam_view)
                # cv2.waitKey(1)
                pass

            

    def cam_start(self):
        self.cam_thread = threading.Thread(target=self.camera_viewer,daemon=True)
        self.cam_thread.start()
        

    def _initInterpolator(self, Arm):
        
        self.interpolator_left = OTG(
            OTG_Dof=self.robot.left_arm.jnt_num,
            control_cycle=0.0005,
            max_velocity=0.05,
            max_acceleration=0.1,
            max_jerk=0.2
        )
        self.interpolator_left.setOTGParam(Arm[0].arm_qpos, Arm[0].arm_qvel)
        self.interpolator_right = OTG(
            OTG_Dof=self.robot.right_arm.jnt_num,
            control_cycle=0.0005,
            max_velocity=0.05,
            max_acceleration=0.1,
            max_jerk=0.2
        )
        self.interpolator_right.setOTGParam(Arm[1].arm_qpos, Arm[1].arm_qvel)

if __name__ == "__main__":


    env = H1_inspireMed(
        robot=H1_inspire(),
        is_render=True,
        control_freq=200,
        cam_view='top',
        is_interpolate=True
    )

    
    env.reset()
    
    for t in range(int(1e6)):
        action_l = np.array([0.0, 0.0, 0.0, 0.0, 0.0 ,0.0, 0.0])
        action_r = np.array([0.0, 0.0, 0.0, 0.0, 0.0 ,0.0, 0.0])
        action = np.hstack((action_l,action_r))
        env.step(action)
        l_q = np.zeros(7)
        l_q[3] = 0.0
        r_q = np.zeros(7)
        r_q[3] = 0.0
        print("\nkdl fk left =",env.arm_left.kdl_solver.getEEtf(l_q))
        # print("\n left pos=",env.robot.robot_data.body("left_link7").xpos,"\n")
        # print("\nleft xmat",env.robot.robot_data.body("left_link7").xmat,"\n")

        print("============================")
        print("\nkdl fk right =",env.arm_right.kdl_solver.getEEtf(r_q))
        # print("\n right pos=",env.robot.robot_data.body("right_link7").xpos,"\n")
        # print("\nright xmat",env.robot.robot_data.body("right_link7").xmat,"\n")

        # quat_left = np.zeros(4)
        # quat_right = np.zeros(4)
        # mj._functions.mju_mat2Quat(quat_left, rot_left)
        # print("quat_left =",quat_left,"\n")
        # mj._functions.mju_mat2Quat(quat_right, rot_right)
        # print("quat_right =",quat_right,"\n")
        if env.is_render:
            env.render()
        