import sys
from collections import deque
import mujoco
from mujoco import viewer
import numpy as np
import time

assert ('linux' in sys.platform)


class MujocoEnv(object):
    """ This environment is the base class.

     :param xml_path(str): Load xml file from xml_path to build the mujoco model.
     :param is_render(bool): Choose if use the renderer to render the scene or not.
     :param renderer(str): choose official renderer with "viewer",
            another renderer with "mujoco_viewer"
     :param control_freq(int): Upper-layer control frequency.
            Note that high frequency will cause high time-lag.
    """

    def __init__(self,
                 robot=None,
                 is_render=False,
                 renderer="viewer",
                 control_freq=1000):

        self.robot = robot
        self.is_render = is_render
        self.renderer = renderer
        self.control_freq = control_freq
        self.mj_model = self.robot.robot_model
        self.mj_data = self.robot.robot_data
        self.viewer = None
        self.cur_time = None
        self.timestep = None
        self.model_timestep = None
        self.control_timestep = None
        self.robot_dof = self.robot.jnt_num
        self.traj = deque(maxlen=200)
        self.render_paused = False
        self.exit_flag = False
        self._rendererInit()
        self._initializeTime(control_freq)
        self.mj_time = time.time()
        # self._setInitPose()

    def step(self, action):#下发电机角度经过PD控制器到mujoco
        """ 
        This method will be called with one-step in mujoco
        :param action: Input action
        :return: None
        """
        self.cur_time += 1
        mujoco.mj_forward(self.mj_model, self.mj_data)#向前计算
        self.preStep(action)#计算并配置电机扭矩
        mujoco.mj_step(self.mj_model, self.mj_data)#步进仿真
        


    def render(self):  #步进rander
        """ render mujoco
        :return: None
        """
        if self.is_render is True and self.viewer is not None:
            if self.renderer == "mujoco_viewer":
                if self.viewer.is_alive is True:
                    self.viewer.render()
                else:
                    sys.exit(0)
            elif self.renderer == "viewer":
                if self.viewer.is_running() and self.exit_flag is False:
                    if(not self.render_paused):
                        self.viewer : viewer.Handle
                        self.viewer.sync()
                else:
                    self.viewer.close()
                    sys.exit(0)

    def _rendererInit(self):  #初始化rander
        """ Initialize renderer, choose official renderer with "viewer"(joined from version 2.3.3),
            another renderer with "mujoco_viewer"
        """
        def key_callback(keycode):
            if keycode == 32:
                self.render_paused = not self.render_paused
            elif keycode == 256:
                self.exit_flag = not self.exit_flag
        if self.is_render is True:
            if self.renderer == "mujoco_viewer":
                import mujoco_viewer
                self.viewer = mujoco_viewer.MujocoViewer(self.mj_model, self.mj_data)
            elif self.renderer == "viewer":
                self.viewer = viewer.launch_passive(self.mj_model, self.mj_data, key_callback=key_callback,show_left_ui = True, show_right_ui=True)

    def _initializeTime(self, control_freq): #根据控制频率来计算控制器下发间隔control_timestep
        """ Initializes the time constants used for simulation.

        :param control_freq (float): Hz rate to run control loop at within the simulation
        """
        self.cur_time = 0
        self.timestep = 0
        self.model_timestep = 0.0005
        self.model_timestep = self.mj_model.opt.timestep
        # print(self.model_timestep)
        if self.model_timestep <= 0:
            raise ValueError("Invalid simulation timestep defined!")
        self.control_freq = control_freq
        if control_freq <= 0:
            raise ValueError("Control frequency {} is invalid".format(control_freq))
        self.control_timestep = 1.0 / control_freq

    def _setInitPose(self): #初始化电机角度
        """ Set or reset init joint position when called env reset func.

        """
        for i in range(len(self.robot.arms)):
            for j in range(len(self.robot.arms[i].joint_index)):
                self.mj_data.joint(self.robot.arms[i].joint_index[j]).qpos = self.robot.arms[i].init_pose[j]
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _setJointPose(self): #控制电机角度
        """ Set or reset init joint position when called env reset func.

        """
        for i in range(len(self.robot.arms)):
            for j in range(len(self.robot.arms[i].joint_index)):
                self.mj_data.joint(self.robot.arms[i].joint_index[j]).qpos = self.robot.arms[i].joint_pose[j]
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def renderTraj(self, pos): #渲染轨迹，版本原因暂时弃用
        """ Render the trajectory from deque above,
            you can push the cartesian position into this deque.

        :param pos: One of the cartesian position of the trajectory to render.
        """
        if self.renderer == "mujoco_viewer" and self.is_render is True:
            if self.cur_time % 10 == 0:
                self.traj.append(pos.copy())
            for point in self.traj:
                self.viewer.add_marker(pos=point, size=np.array([0.001, 0.001, 0.001]), rgba=np.array([0, 0, 1, 1]),
                                       type=mujoco.mjtGeom.mjGEOM_SPHERE)

    def preStep(self, action):#把电机角度经过PD控制器转换为扭矩，具体实现在子类中
        """ Writes your own codes between mujoco forward and step you want to control.

        :param action: input actions
        :return: None
        """
        raise NotImplementedError

    def getName2ID(self, mj_type : str, name: str):
        if mj_type == 'geom':
            input_type = mujoco.mjtObj.mjOBJ_GEOM
        elif mj_type == 'body':
            input_type = mujoco.mjtObj.mjOBJ_BODY
        elif mj_type == 'camera':
            input_type = mujoco.mjtObj.mjOBJ_CAMERA
        elif mj_type == 'joint':
            input_type = mujoco.mjtObj.mjOBJ_JOINT
        else:
            raise AttributeError("input geom, body, joint, or camera")
        return mujoco.mj_name2id(self.mj_model, input_type, name)

    def getID2Name(self, mj_type : str, id : int):
        if mj_type == 'geom':
            input_type = mujoco.mjtObj.mjOBJ_GEOM
        elif mj_type == 'body':
            input_type = mujoco.mjtObj.mjOBJ_BODY
        elif mj_type == 'camera':
            input_type = mujoco.mjtObj.mjOBJ_CAMERA
        elif mj_type == 'joint':
            input_type = mujoco.mjtObj.mjOBJ_JOINT
        else:
            raise AttributeError("input geom, body, joint, or camera")
        return mujoco.mj_id2name(self.mj_model,input_type, id)


    def getBodyPos(self, name : str):
        return self.mj_data.body(name).xpos
    
    def getBodyMat(self, name : str):
        xmat = np.array(self.mj_data.body(name).xmat, dtype=np.float32).reshape(3, 3)
        return xmat
    
    def getBodyQuat(self, name : str):
        return self.mj_data.body(name).xquat

    def reset(self): #重置mujoco环境
        """ Reset the simulate environment, in order to execute next episode.

        """
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self._setInitPose()
        mujoco.mj_step(self.mj_model, self.mj_data)

    
