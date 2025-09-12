from assets.robots.arm_base import *
import os
import numpy as np
from numpy.linalg import norm, solve
import numpy as np
import pathlib
HOME_PATH = str(pathlib.Path("../").parent.resolve())
import sys
sys.path.append(HOME_PATH)


class H1_inspire(ArmBase): #定义两个机械臂的控制接口
    def __init__(self):
        super().__init__(
            name="h1_inspire",
            urdf_path=HOME_PATH+"/Mujoco_env/assets/models/manipulators/h1_2/h1_2.urdf",
            xml_path=HOME_PATH+"/Mujoco_env/assets/models/manipulators/h1_2/scene.xml",
            gripper=None
        )
        self.arms = []
        self.left_arm = self.Arm(self, 'single', self.urdf_path)
        self.left_arm.set_Arm_base_link('torso_link')
        self.left_arm.set_Arm_ee_link('left_wrist_yaw_link')
        self.left_arm.InitKDL
        self.left_arm.joint_index = ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_pitch_joint', 'left_elbow_roll_joint','left_wrist_pitch_joint','left_wrist_yaw_joint']
        # self.left_arm.gripper_index = ['L_l_finger_joint','L_r_finger_joint']
        self.left_arm.actuator_index = ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_pitch_joint', 'left_elbow_roll_joint','left_wrist_pitch_joint','left_wrist_yaw_joint']

        self.left_arm.setArmInitPose(self.init_qpos)
        # self.left_arm.setArmJointPose(self.init_qpos)
        self.arms.append(self.left_arm)
        self.right_arm = self.Arm(self,'single', self.urdf_path)
        self.right_arm.set_Arm_base_link('torso_link')
        self.right_arm.set_Arm_ee_link('right_wrist_yaw_link')
        self.right_arm.InitKDL
        self.right_arm.joint_index = ['right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_pitch_joint', 'right_elbow_roll_joint','right_wrist_pitch_joint','right_wrist_yaw_joint']
        # self.right_arm.gripper_index = ['R_l_finger_joint','R_r_finger_joint']
        self.right_arm.actuator_index = ['right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_pitch_joint', 'right_elbow_roll_joint','right_wrist_pitch_joint','right_wrist_yaw_joint']
        self.right_arm.setArmInitPose(self.init_qpos)
        # self.right_arm.setArmJointPose(self.init_qpos)
        self.arms.append(self.right_arm)


        #腰
        self.torso=self.Arm(self, 'single', self.urdf_path)
        self.torso.set_Arm_base_link('pelvis')
        self.torso.set_Arm_ee_link('torso_link')
        self.torso.InitKDL
        self.torso.joint_index = ['torso_joint']
        self.torso.actuator_index = ['torso_joint']
        self.torso.setArmInitPose(np.zeros(1))
        self.arms.append(self.torso)

        #左手大拇指
        self.l_hand1=self.Arm(self, 'single', self.urdf_path)
        self.l_hand1.set_Arm_base_link('L_hand_base_link')
        self.l_hand1.set_Arm_ee_link('L_thumb_distal')
        self.l_hand1.InitKDL
        self.l_hand1.joint_index = ['L_thumb_proximal_yaw_joint','L_thumb_proximal_pitch_joint','L_thumb_intermediate_joint','L_thumb_distal_joint']
        self.l_hand1.actuator_index = ['L_thumb_proximal_yaw_joint','L_thumb_proximal_pitch_joint','L_thumb_intermediate_joint','L_thumb_distal_joint']
        self.l_hand1.setArmInitPose(np.zeros(4))
        self.arms.append(self.l_hand1)
        #左手食指
        self.l_hand2=self.Arm(self, 'single', self.urdf_path)
        self.l_hand2.set_Arm_base_link('L_hand_base_link')
        self.l_hand2.set_Arm_ee_link('L_index_intermediate')
        self.l_hand2.InitKDL
        self.l_hand2.joint_index = ['L_index_proximal_joint','L_index_intermediate_joint']
        self.l_hand2.actuator_index = ['L_index_proximal_joint','L_index_intermediate_joint']
        self.l_hand2.setArmInitPose(np.zeros(2))
        self.arms.append(self.l_hand2)  
        #左手中指
        self.l_hand3=self.Arm(self, 'single', self.urdf_path)
        self.l_hand3.set_Arm_base_link('L_hand_base_link')
        self.l_hand3.set_Arm_ee_link('L_middle_intermediate')
        self.l_hand3.InitKDL
        self.l_hand3.joint_index = ['L_middle_proximal_joint','L_middle_intermediate_joint']
        self.l_hand3.actuator_index = ['L_middle_proximal_joint','L_middle_intermediate_joint']
        self.l_hand3.setArmInitPose(np.zeros(2))
        self.arms.append(self.l_hand3)  
        #左手无名指
        self.l_hand4=self.Arm(self, 'single', self.urdf_path)
        self.l_hand4.set_Arm_base_link('L_hand_base_link')
        self.l_hand4.set_Arm_ee_link('L_ring_intermediate')
        self.l_hand4.InitKDL
        self.l_hand4.joint_index = ['L_ring_proximal_joint','L_ring_intermediate_joint']
        self.l_hand4.actuator_index = ['L_ring_proximal_joint','L_ring_intermediate_joint']
        self.l_hand4.setArmInitPose(np.zeros(2))
        self.arms.append(self.l_hand4)  
        #左手小拇指
        self.l_hand5=self.Arm(self, 'single', self.urdf_path)
        self.l_hand5.set_Arm_base_link('L_hand_base_link')
        self.l_hand5.set_Arm_ee_link('L_pinky_intermediate')
        self.l_hand5.InitKDL
        self.l_hand5.joint_index = ['L_pinky_proximal_joint','L_pinky_intermediate_joint']
        self.l_hand5.actuator_index = ['L_pinky_proximal_joint','L_pinky_intermediate_joint']
        self.l_hand5.setArmInitPose(np.zeros(2))
        self.arms.append(self.l_hand5) 


        #右手大拇指
        self.r_hand1=self.Arm(self, 'single', self.urdf_path)
        self.r_hand1.set_Arm_base_link('R_hand_base_link')
        self.r_hand1.set_Arm_ee_link('R_thumb_distal')
        self.r_hand1.InitKDL
        self.r_hand1.joint_index = ['R_thumb_proximal_yaw_joint','R_thumb_proximal_pitch_joint','R_thumb_intermediate_joint','R_thumb_distal_joint']
        self.r_hand1.actuator_index = ['R_thumb_proximal_yaw_joint','R_thumb_proximal_pitch_joint','R_thumb_intermediate_joint','R_thumb_distal_joint']
        self.r_hand1.setArmInitPose(np.zeros(4))
        self.arms.append(self.r_hand1)
        #右手食指
        self.r_hand2=self.Arm(self, 'single', self.urdf_path)
        self.r_hand2.set_Arm_base_link('R_hand_base_link')
        self.r_hand2.set_Arm_ee_link('R_index_intermediate')
        self.r_hand2.InitKDL
        self.r_hand2.joint_index = ['R_index_proximal_joint','R_index_intermediate_joint']
        self.r_hand2.actuator_index = ['R_index_proximal_joint','R_index_intermediate_joint']
        self.r_hand2.setArmInitPose(np.zeros(2))
        self.arms.append(self.r_hand2)  
        #右手中指
        self.r_hand3=self.Arm(self, 'single', self.urdf_path)
        self.r_hand3.set_Arm_base_link('R_hand_base_link')
        self.r_hand3.set_Arm_ee_link('R_middle_intermediate')
        self.r_hand3.InitKDL
        self.r_hand3.joint_index = ['R_middle_proximal_joint','R_middle_intermediate_joint']
        self.r_hand3.actuator_index = ['R_middle_proximal_joint','R_middle_intermediate_joint']
        self.r_hand3.setArmInitPose(np.zeros(2))
        self.arms.append(self.r_hand3)  
        #右手无名指
        self.r_hand4=self.Arm(self, 'single', self.urdf_path)
        self.r_hand4.set_Arm_base_link('R_hand_base_link')
        self.r_hand4.set_Arm_ee_link('R_ring_intermediate')
        self.r_hand4.InitKDL
        self.r_hand4.joint_index = ['R_ring_proximal_joint','R_ring_intermediate_joint']
        self.r_hand4.actuator_index = ['R_ring_proximal_joint','R_ring_intermediate_joint']
        self.r_hand4.setArmInitPose(np.zeros(2))
        self.arms.append(self.r_hand4)  
        #右手小拇指
        self.r_hand5=self.Arm(self, 'single', self.urdf_path)
        self.r_hand5.set_Arm_base_link('R_hand_base_link')
        self.r_hand5.set_Arm_ee_link('R_pinky_intermediate')
        self.r_hand5.InitKDL
        self.r_hand5.joint_index = ['R_pinky_proximal_joint','R_pinky_intermediate_joint']
        self.r_hand5.actuator_index = ['R_pinky_proximal_joint','R_pinky_intermediate_joint']
        self.r_hand5.setArmInitPose(np.zeros(2))
        self.arms.append(self.r_hand5) 


        self.jnt_num = (
            self.left_arm.jnt_num + self.right_arm.jnt_num
            +self.torso.jnt_num
            +self.l_hand1.jnt_num+self.l_hand2.jnt_num+self.l_hand3.jnt_num+self.l_hand4.jnt_num+self.l_hand5.jnt_num
            +self.r_hand1.jnt_num+self.r_hand2.jnt_num+self.r_hand3.jnt_num+self.r_hand4.jnt_num+self.r_hand5.jnt_num
        )

        # joint PD
        self.kp = np.ones(7)*700
        self.kd = np.ones(7)*260

        print("H1 init !")





    @property
    def init_qpos(self):
        """ Robot's init joint position. """
        return np.array([0.0, 0.0, 0.0, 0, 0.0, 0, 0])
    
