import os
import numpy as np
import PyKDL as kdl
from urdf_parser_py.urdf import URDF
from Mujoco_env.utils_.KDL_utils import KDL_main


class KDL_utils:

    def __init__(self,
                 urdf_path=os.path.join(os.path.dirname(__file__), "../images/DianaMed/urdf/DianaMed.urdf")):
        # Build kdl chain
        urdf = URDF.from_xml_file(urdf_path)
        self.tree = KDL_main.kdl_tree_from_urdf_model(urdf)
        self.env_grav = kdl.Vector(0.0, 0.0, -9.81)  # default
        self.ee_frame = kdl.Frame()
        self.rot_general = kdl.Rotation()
        

    def resetChain(self, base, end):
        self.chain = self.tree.getChain(base, end)  # default for DianaMed

        # Initialize solvers
        self.ik_solver = kdl.ChainIkSolverPos_LMA(self.chain, eps=0.001, maxiter=200, eps_joints=0.0)
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.fk_solvervel = kdl.ChainFkSolverVel_recursive(self.chain)
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)
        self.jacdot_solver = kdl.ChainJntToJacDotSolver(self.chain)
        self.diff_ik_solver = kdl.ChainIkSolverVel_pinv(self.chain, eps=0.00001, maxiter=200)

        # Params
        self.NbOfJnt = self.chain.getNrOfJoints()
        self._dyn_kdl = kdl.ChainDynParam(self.chain, self.env_grav)
        self.mass_kdl = kdl.JntSpaceInertiaMatrix(self.NbOfJnt)
        self.corio_kdl = kdl.JntArray(self.NbOfJnt)
        self.grav_kdl = kdl.JntArray(self.NbOfJnt)
        self.ee_frame = kdl.Frame()
        

    def setJntArray(self, q):
        # print(self.NbOfJnt)
        q_kdl = kdl.JntArray(self.NbOfJnt)
        for i in range(len(q)):
            q_kdl[i] = q[i]
        return q_kdl
    
    def setJntArrayVel(self, q, qdot):
        q_kdl = kdl.JntArray(self.NbOfJnt)
        qdot_kdl = kdl.JntArray(self.NbOfJnt)
        for i in range(len(q)):
            # for i in range(7):
            q_kdl[i] = q[i]
            qdot_kdl[i] = qdot[i]
        return kdl.JntArrayVel(q_kdl,qdot_kdl)

    def setNumpyMat(self, mat):
        if isinstance(mat, kdl.Rotation):
            m = np.zeros((3, 3))
        else:
            m = np.zeros((mat.rows(), mat.columns()))
            # m = np.zeros((7, 7))
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                m[i, j] = mat[i, j]
        return m

    def setRot(self,mat : np.ndarray):
        if len(mat) == 9:
            rot = kdl.Rotation()
            for i in range(3):
                for j in range(3):
                    rot[i,j] = mat[i*3+j]
            return rot
        else: 
            raise AttributeError("need numpy array size = 9 !")
    
    def setVec(self,array : np.ndarray):
        if len(array) == 3:
            vec = kdl.Vector()
            for i in range(3):
                vec[i] = array[i]
            return vec
        else:
            print("len input=",len(array))
            raise AttributeError("need numpy array size = 3 !")

    def setNumpyArray(self, q):
        return np.asarray([q[i] for i in range(q.rows())], dtype=np.float32)

    def getInertiaMat(self, q):
        _q = self.setJntArray(q)
        self._dyn_kdl.JntToMass(_q, self.mass_kdl)
        return self.setNumpyMat(self.mass_kdl)

    def getCoriolisMat(self, q, qdot):
        _q = self.setJntArray(q)
        _qdot = self.setJntArray(qdot)
        self._dyn_kdl.JntToCoriolis(_q, _qdot, self.corio_kdl)
        return self.corio_kdl

    def getGravityMat(self, q):
        _q = self.setJntArray(q)
        self._dyn_kdl.JntToGravity(_q, self.grav_kdl)
        return self.grav_kdl

    def getCompensation(self, q, qdot):
        comp = np.zeros(7)
        self._dyn_kdl.JntToCoriolis(q, qdot, self.corio_kdl)
        self._dyn_kdl.JntToGravity(q, self.grav_kdl)
        for i in range(self.NbOfJnt):
            comp[i] = self.corio_kdl[i] + self.grav_kdl[i]
        return comp

    def getJac(self, q=kdl.JntArray()):
        j_ = kdl.Jacobian(self.NbOfJnt)
        self.jac_solver.JntToJac(q, j_)
        return j_

    def getJac_pinv(self, q=kdl.JntArray()):
        j_ = self.getJac(q)
        print("shape==",j_.shape)
        jacobian = np.empty((6, 7))
        for row in range(6):
            for col in range(7):
                jacobian[row][col] = j_.getColumn(col)[row]
        jacobian_pinv = np.linalg.pinv(jacobian)
        return jacobian_pinv


    def getEEtf(self, joint_states):
        joint_states = self.setJntArray(joint_states)
        ee_frame = kdl.Frame()
        self.fk_solver.JntToCart(joint_states, ee_frame, self.NbOfJnt)
        
        return ee_frame

    def getEeCurrentPose(self, q):
        """ Get current pose with input's joint state.

        :param q: Current joint states
        :return: Current position and rotation
        """
        self.fk_solver.JntToCart(self.setJntArray(q), self.ee_frame, self.NbOfJnt)
        pos = np.array([self.ee_frame.p[i] for i in range(3)])
        rot = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                rot[i][j] = self.ee_frame.M[i, j]
        return pos, rot

    def fkSolverVel(self, qdot_init : kdl.JntArrayVel=None):
        if qdot_init is None:
            raise NotImplementedError
        cart = kdl.FrameVel().Identity()
        self.fk_solvervel.JntToCart(qdot_init,cart)
        return cart.deriv()
        
    def JntArray(self,dof):
        return kdl.JntArray(dof)


    def ikSolver(self, pos: np.ndarray, rot: np.ndarray, q_init: np.ndarray = None) -> np.ndarray:
        if q_init is None:
            raise NotImplementedError
        else:
            q_init = self.setJntArray(q_init)

            for i in range(3):
                self.ee_frame.p[i] = pos[i]
                for j in range(3):
                    self.ee_frame.M[i, j] = rot[i][j]
            q_result = kdl.JntArray(7)
            status=self.ik_solver.CartToJnt(q_init, self.ee_frame, q_result)
            # print(status)
            return self.setNumpyArray(q_result)

    def diffikSolver(self, q_init : np.ndarray, v_init : np.ndarray):
        if  (q_init is None) or (v_init is None):
            raise NotImplementedError
        else:
            q_init = self.setJntArray(q_init)
            v_init = kdl.Twist(self.setVec(v_init[:3]),self.setVec(v_init[3:]))
            qdot_result = kdl.JntArray(7)
            self.diff_ik_solver.CartToJnt(q_init,v_init,qdot_result)
            return self.setNumpyArray(qdot_result)


    def jacobian_dot(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """

        Args:
            q (np.ndarray): current joint position  
            v (np.ndarray): current joint velocity

        Returns:
            np.ndarray: jacobian_dot
        """
        input_q = self.setJntArray(q)
        input_qd = self.setJntArray(v)
        input_qav = kdl.JntArrayVel(input_q, input_qd)
        output = kdl.Jacobian(self.NbOfJnt)
        self.jacdot_solver.JntToJacDot(input_qav, output)
        return self.setNumpyMat(output)

    def orientation_error(self, desired: np.ndarray, current: np.ndarray) -> np.ndarray:
        """computer ori error from ori to cartesian 姿态矩阵的偏差3*3的
        Args:
            desired (np.ndarray): desired orientation
            current (np.ndarray): current orientation

        Returns:
            _type_: orientation error(from pose(3*3) to eulor angular(3*1))
        """
        rc1 = current[:, 0]
        rc2 = current[:, 1]
        rc3 = current[:, 2]
        rd1 = desired[:, 0]
        rd2 = desired[:, 1]
        rd3 = desired[:, 2]
        if (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3)).all() <= 0.0001:
            w1, w2, w3 = 0.5, 0.5, 0.5
        else:
            w1, w2, w3 = 0.9, 0.5, 0.3

        error = w1 * np.cross(rc1, rd1) + w2 * np.cross(rc2, rd2) + w3 * np.cross(rc3, rd3)

        return error
    
    def xmat2Rotation(self,xmat : np.ndarray):
        '''
        Args:
            xmat (type: np.ndarray): the rotation matrix got from mujoco
        Returns:
            rot (KDL Rotation()): which get the value from xmat(3*3)
        '''
        if len(xmat) == 9 :
            rot = kdl.Rotation()
            rot[0,0] = xmat[0]
            rot[1,0] = xmat[1]
            rot[2,0] = xmat[2]
            rot[0,1] = xmat[3]
            rot[1,1] = xmat[4]
            rot[2,1] = xmat[5]
            rot[2,0] = xmat[6]
            rot[2,1] = xmat[7]
            rot[2,2] = xmat[8]
            return rot
        else:
            print("please entry xmat from mujoco")
            return 0
        
    def xpos2Vector(self,xpos : np.ndarray):
        pos_vec = kdl.Vector()
        for i in range(len(xpos)):
            pos_vec[i] = xpos[i]
        return pos_vec
    
    def jac_mj2Jac_KDL(self):
        pass
        
        
class KDL_utils_X02:

    def __init__(self,
                 urdf_path=os.path.join(os.path.dirname(__file__),
                                         "../images/DianaMed/urdf/DianaMed.urdf")):
        # Build kdl chain
        urdf = URDF.from_xml_file(urdf_path)
        self.tree = KDL_main.kdl_tree_from_urdf_model(urdf)
        self.chain = self.tree.getChain("base_link", "L_wrist3_Link")  # default for DianaMed

        # Initialize solvers
        self.ik_solver = kdl.ChainIkSolverPos_LMA(self.chain, eps=0.0, maxiter=200, eps_joints=0.0)
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)
        self.jacdot_solver = kdl.ChainJntToJacDotSolver(self.chain)

        # Params
        self.NbOfJnt = self.chain.getNrOfJoints()
        self.env_grav = kdl.Vector(0.0, 0.0, -9.81)  # default
        self._dyn_kdl = kdl.ChainDynParam(self.chain, self.env_grav)
        self.mass_kdl = kdl.JntSpaceInertiaMatrix(self.NbOfJnt)
        self.corio_kdl = kdl.JntArray(self.NbOfJnt)
        self.grav_kdl = kdl.JntArray(self.NbOfJnt)
        self.ee_frame = kdl.Frame()
        self.rot_general = kdl.Rotation()
        

    def resetChain(self, base, end):
        self.chain = self.tree.getChain(base, end)  # default for DianaMed

        # Initialize solvers
        self.ik_solver = kdl.ChainIkSolverPos_LMA(self.chain, eps=0.0, maxiter=200, eps_joints=0.0)
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)
        self.jacdot_solver = kdl.ChainJntToJacDotSolver(self.chain)

        # Params
        self.NbOfJnt = self.chain.getNrOfJoints()
        self.env_grav = kdl.Vector(0.0, 0.0, -9.81)  # default
        self._dyn_kdl = kdl.ChainDynParam(self.chain, self.env_grav)
        self.mass_kdl = kdl.JntSpaceInertiaMatrix(self.NbOfJnt)
        self.corio_kdl = kdl.JntArray(self.NbOfJnt)
        self.grav_kdl = kdl.JntArray(self.NbOfJnt)
        self.ee_frame = kdl.Frame()
        

    def setJntArray(self, q):
        q_kdl = kdl.JntArray(self.NbOfJnt)
        for i in range(len(q)):
            # for i in range(7):
            q_kdl[i] = q[i]
        return q_kdl

    def setNumpyMat(self, mat):
        if isinstance(mat, kdl.Rotation):
            m = np.zeros((3, 3))
        else:
            m = np.zeros((mat.rows(), mat.columns()))
            # m = np.zeros((7, 7))
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                m[i, j] = mat[i, j]
        return m

    def setNumpyArray(self, q):
        return np.asarray([q[i] for i in range(q.rows())], dtype=np.float32)

    def getInertiaMat(self, q):
        _q = self.setJntArray(q)
        self._dyn_kdl.JntToMass(_q, self.mass_kdl)
        return self.setNumpyMat(self.mass_kdl)

    def getCoriolisMat(self, q, qdot):
        _q = self.setJntArray(q)
        _qdot = self.setJntArray(qdot)
        self._dyn_kdl.JntToCoriolis(_q, _qdot, self.corio_kdl)
        return self.corio_kdl

    def getGravityMat(self, q):
        _q = self.setJntArray(q)
        self._dyn_kdl.JntToGravity(_q, self.grav_kdl)
        return self.grav_kdl

    def getCompensation(self, q, qdot):
        comp = np.zeros(7)
        self._dyn_kdl.JntToCoriolis(q, qdot, self.corio_kdl)
        self._dyn_kdl.JntToGravity(q, self.grav_kdl)
        for i in range(self.NbOfJnt):
            comp[i] = self.corio_kdl[i] + self.grav_kdl[i]
        return comp

    def getJac(self, q=kdl.JntArray()):
        j_ = kdl.Jacobian(self.NbOfJnt)
        self.jac_solver.JntToJac(q, j_)
        return j_

    def getJac_pinv(self, q=kdl.JntArray()):
        j_ = self.getJac(q)
        jacobian = np.empty((6, 7))
        for row in range(6):
            for col in range(7):
                jacobian[row][col] = j_.getColumn(col)[row]
        jacobian_pinv = np.linalg.pinv(jacobian)
        return jacobian_pinv

    def getEEtf(self, joint_states):
        ee_frame = kdl.Frame()
        self.fk_solver.JntToCart(joint_states, ee_frame, self.NbOfJnt)
        return ee_frame

    def getEeCurrentPose(self, q):
        """ Get current pose with input's joint state.

        :param q: Current joint states
        :return: Current position and rotation
        """
        self.fk_solver.JntToCart(self.setJntArray(q), self.ee_frame, self.NbOfJnt)
        pos = np.array([self.ee_frame.p[i] for i in range(3)])
        rot = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                rot[i][j] = self.ee_frame.M[i, j]
        return pos, rot

    def ikSolver(self, pos: np.ndarray, rot: np.ndarray, q_init: np.ndarray = None) -> np.ndarray:
        if q_init is None:
            raise NotImplementedError
        q_init = self.setJntArray(q_init)

        for i in range(3):
            self.ee_frame.p[i] = pos[i]
            for j in range(3):
                self.ee_frame.M[i, j] = rot[i][j]
        q_result = kdl.JntArray(7)
        self.ik_solver.CartToJnt(q_init, self.ee_frame, q_result)
        return self.setNumpyArray(q_result)

    def jacobian_dot(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """

        Args:
            q (np.ndarray): current joint position  
            v (np.ndarray): current joint velocity

        Returns:
            np.ndarray: jacobian_dot
        """
        input_q = self.setJntArray(q)
        input_qd = self.setJntArray(v)
        input_qav = kdl.JntArrayVel(input_q, input_qd)
        output = kdl.Jacobian(self.NbOfJnt)
        self.jacdot_solver.JntToJacDot(input_qav, output)
        return self.setNumpyMat(output)

    def orientation_error(self, desired: np.ndarray, current: np.ndarray) -> np.ndarray:
        """computer ori error from ori to cartesian 姿态矩阵的偏差3*3的
        Args:
            desired (np.ndarray): desired orientation
            current (np.ndarray): current orientation

        Returns:
            _type_: orientation error(from pose(3*3) to eulor angular(3*1))
        """
        rc1 = current[:, 0]
        rc2 = current[:, 1]
        rc3 = current[:, 2]
        rd1 = desired[:, 0]
        rd2 = desired[:, 1]
        rd3 = desired[:, 2]
        if (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3)).all() <= 0.0001:
            w1, w2, w3 = 0.5, 0.5, 0.5
        else:
            w1, w2, w3 = 0.9, 0.5, 0.3

        error = w1 * np.cross(rc1, rd1) + w2 * np.cross(rc2, rd2) + w3 * np.cross(rc3, rd3)

        return error
    
    def xmat2Rotation(self,xmat : np.ndarray):
        '''
        Args:
            xmat (type: np.ndarray): the rotation matrix got from mujoco
        Returns:
            rot (KDL Rotation()): which get the value from xmat(3*3)
        '''
        if len(xmat) == 9 :
            rot = kdl.Rotation()
            rot[0,0] = xmat[0]
            rot[1,0] = xmat[1]
            rot[2,0] = xmat[2]
            rot[0,1] = xmat[3]
            rot[1,1] = xmat[4]
            rot[2,1] = xmat[5]
            rot[2,0] = xmat[6]
            rot[2,1] = xmat[7]
            rot[2,2] = xmat[8]
            return rot
        else:
            print("please entry xmat from mujoco")
            return 0
        
    def xpos2Vector(self,xpos : np.ndarray):
        pos_vec = kdl.Vector()
        for i in range(len(xpos)):
            pos_vec[i] = xpos[i]
        return pos_vec
    
    def jac_mj2Jac_KDL(self):
        pass