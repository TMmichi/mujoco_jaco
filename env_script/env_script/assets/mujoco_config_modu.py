from xml.etree import ElementTree

import os
import numpy as np

import mujoco_py as mjp


class MujocoConfig:
    """ A wrapper on the Mujoco simulator to generate all the kinematics and
    dynamics calculations necessary for controllers.
    """

    def __init__(self, xml_file, folder=None, use_sim_state=True, force_download=False):
        """ Loads the Mujoco model from the specified xml file

        Parameters
        ----------
        xml_file: string
            the name of the arm model to load. If folder remains as None,
            the string passed in is parsed such that everything up to the first
            underscore is used for the arm directory, and the full string is
            used to load the xml within that folder.

            EX: 'myArm' and 'myArm_with_gripper' will both look in the
            'myArm' directory, however they will load myArm.xml and
            myArm_with_gripper.xml, respectively

            If a folder is passed in, then folder/xml_file is used
        folder: string, Optional (Default: None)
            specifies what folder to find the xml_file, if None specified will
            checking in abr_control/arms/xml_file (see above for xml_file)
        use_sim_state: Boolean, optional (Default: True)
            If set true, the q and dq values passed in to the functions are
            ignored, and the current state of the simulator is used to
            calculate all functions. Can speed up simulation by not resetting
            the state on every call.
        """

        if folder is None:
            arm_dir = xml_file.split("_")[0]
            current_dir = os.path.dirname(__file__)
            try:
                self.ctrl_type = xml_file.split("_")[1]
                self.xml_file = os.path.join(current_dir, arm_dir, "%s.xml" % xml_file)
            except Exception:
                self.ctrl_type = "torque"
                self.xml_file = os.path.join(current_dir, arm_dir, "%s_torque.xml" % xml_file)
            assert self.ctrl_type in ["torque", "velocity", "position"], "[ERROR] Wrong Control Type: " + \
                "Control type of your robot should be one of the following - 'torque', 'velocity', 'position'"
            self.xml_dir = "%s/%s" % (current_dir, arm_dir)

        self.N_GRIPPEPR_JOINTS = 0

        # get access to some of our custom arm parameters from the xml definition
        tree = ElementTree.parse(self.xml_file)
        root = tree.getroot()
        for custom in root.findall("custom/numeric"):
            name = custom.get("name")
            if name == "START_ANGLES":
                START_ANGLES = custom.get("data").split(" ")
                self.START_ANGLES = np.array([float(angle) for angle in START_ANGLES])
            elif name == "N_GRIPPER_JOINTS":
                self.N_GRIPPER_JOINTS = int(custom.get("data"))

        self.model = mjp.load_model_from_path(self.xml_file)
        self.use_sim_state = use_sim_state


    def _connect(self, sim, joint_pos_addrs, joint_vel_addrs, joint_dyn_addrs):
        """ Called by the interface once the Mujoco simulation is created,
        this connects the config to the simulator so it can access the
        kinematics and dynamics information calculated by Mujoco.

        Parameters
        ----------
        sim: MjSim
            The Mujoco Simulator object created by the Mujoco Interface class
        joint_pos_addrs: np.array of ints
            The index of the robot joints in the Mujoco simulation data joint
            position array
        joint_vel_addrs: np.array of ints
            The index of the robot joints in the Mujoco simulation data joint
            velocity array
        joint_dyn_addrs: np.array of ints
            The index of the robot joints in the Mujoco simulation data joint
            Jacobian, inertia matrix, and gravity vector
        """
        # get access to the Mujoco simulation
        self.sim = sim
        self.joint_pos_addrs = np.copy(joint_pos_addrs)
        self.joint_vel_addrs = np.copy(joint_vel_addrs)
        self.joint_dyn_addrs = np.copy(joint_dyn_addrs)

        # number of joints in the robot arm
        self.N_JOINTS = len(self.joint_pos_addrs)
        # number of joints in the Mujoco simulation
        N_ALL_JOINTS = int(len(self.sim.data.get_body_jacp("EE")) / 3)

        # need to calculate the joint_dyn_addrs indices in flat vectors returned
        # for the Jacobian
        self.jac_indices = np.hstack(
            # 6 because position and rotation Jacobians are 3 x N_JOINTS
            [self.joint_dyn_addrs + (ii * N_ALL_JOINTS) for ii in range(3)]
        )

        # for the inertia matrix
        self.M_indices = [
            ii * N_ALL_JOINTS + jj
            for jj in self.joint_dyn_addrs
            for ii in self.joint_dyn_addrs
        ]

        # a place to store data returned from Mujoco
        self._g = np.zeros(self.N_JOINTS)
        self._J3NP = np.zeros(3 * N_ALL_JOINTS)
        self._J3NR = np.zeros(3 * N_ALL_JOINTS)
        self._J6N = np.zeros((6, self.N_JOINTS))
        self._MNN_vector = np.zeros(N_ALL_JOINTS ** 2)
        self._MNN = np.zeros(self.N_JOINTS ** 2)
        self._R9 = np.zeros(9)
        self._R = np.zeros((3, 3))
        self._x = np.ones(4)
        self.N_ALL_JOINTS = N_ALL_JOINTS

    def quaternion(self, name, q=None):
        """ Returns the quaternion of the specified body
        Parameters
        ----------

        name: string
            The name of the Mujoco body to retrieve the Jacobian for
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        """
        quaternion = np.copy(self.sim.data.get_body_xquat(name))
        return quaternion