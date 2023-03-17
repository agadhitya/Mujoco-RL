from collections import OrderedDict
import numpy as np

from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, MujocoXMLObject, CylinderObject
from robosuite.models.tasks import ManipulationTask

from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string
from robosuite.utils.mjcf_utils import CustomMaterial, find_elements

class InsertionHoleObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(xml_path_completion("objects/insertion_hole1.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

class FixedInsertionHoleObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(xml_path_completion("objects/insertion_hole1.xml"),
                         name=name, joints=None,
                         obj_type="all", duplicate_collision_geoms=True)

class Insert(SingleArmEnv):
    """
    This class corresponds to the insertion task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (hole) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise=None, #"default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=True, #Was False
        peg_radius=(0.020, 0.020), #(0.024, 0.024)
        peg_length=0.06,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera='frontview', #None/frontview
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1024,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))
        self.table_z=self.table_offset[2]+table_full_size[2]

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # Save peg specs
        self.peg_radius = peg_radius
        self.peg_length = peg_length

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 5.0 is provided if the peg is inside the plate's hole
              - Note that we enforce that it's inside at an appropriate angle (cos(theta) > 0.95).

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arms to approach each other
            - Perpendicular Distance: in [0,1], to encourage the arms to approach each other
            - Parallel Distance: in [0,1], to encourage the arms to approach each other
            - Alignment: in [0, 1], to encourage having the right orientation between the peg and hole.
            - Placement: in {0, 1}, nonzero if the peg is in the hole with a relatively correct alignment

        Note that the final reward is normalized and scaled by reward_scale / 5.0 as
        well so that the max score is equal to reward_scale
        """
        reward = 0.

        #Get peg position (known from joint encoders)
        peg_pos = self.sim.data.body_xpos[self.peg_body_id]
        hole_pos = self.sim.data.body_xpos[self.hole_body_id]


        #Calculate distance from peg to 'bottom' surface of hole
        dist = np.linalg.norm(peg_pos - hole_pos)
        z_dist=peg_pos[2]-(self.table_z+0.085)
        #print('Dist:'+str(dist)+', Fz:'+str(self.robots[0].ee_force[2]))

        #For just z-component (threshold=0.01). For 3D (threshold=0.086)
        threshold=0.086
        if (dist<threshold) and (dist>-0.00001) and (self.robots[0].ee_force[2]<-0.5):
            reward=1.0

        #If peg is too low or episode is ending, penalize
        if (z_dist<-0.00001) or (self.timestep >= self.horizon):
            reward=-1.0

        #TODO: implement exp decay with negative rewards after a while
        #If robot has inserted peg into hole
        if reward>0:
            #Hack for scaling rewards to reduce time required
            #reward*=((self.max_t-t)/self.max_t) #linear decay
            reward*=(0.02**(self.timestep/self.horizon)) #exp. decay

        return reward, z_dist

    def _post_action(self, action):
        """
        Do any housekeeping after taking an action.

        Args:
            action (np.array): Action to execute within the environment

        Returns:
            3-tuple:

                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) empty dict to be filled with information by subclassed method

        """
        reward, z_dist = self.reward(action)

        # done if number of elapsed timesteps is greater than horizon
        if not self.ignore_done:
            self.done = (self.timestep >= self.horizon) or (reward>0) or (z_dist<-0.00001)

        return reward, self.done, {}

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.hole = InsertionHoleObject(
            name="free_insert_hole",
        )

        self.peg = CylinderObject(
            name="peg",
            size_min=(self.peg_radius[0], self.peg_length),
            size_max=(self.peg_radius[1], self.peg_length),
            material=greenwood,
            rgba=[0, 1, 0, 1],
            joints=None,
            #TODO: DEFINE THESE
            density=None,
            friction=None,
            solref=None,
            solimp=None,
        )

        # Create placement initializer for hole
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.hole)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.hole,
                x_range=[-0.01,0.01],
                y_range=[-0.01,0.01],
                rotation=0.0, #None,
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.0, #0.01
            )

        # Load peg object
        peg_obj = self.peg.get_obj()
        peg_obj.set("pos", array_to_string((0, 0, 2.5*self.peg_length)))
        r_eef = self.robots[0].robot_model.eef_name
        r_model = self.robots[0].robot_model
        r_body = find_elements(root=r_model.worldbody, tags="body", attribs={"name": r_eef}, return_first=True)
        r_body.append(peg_obj)

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.hole,
        )

        self.model.merge_assets(self.peg)

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.hole_body_id = self.sim.model.body_name2id(self.hole.root_body)
        self.peg_body_id = self.sim.model.body_name2id(self.peg.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # FT sensor
            @sensor(modality=modality)
            def ft_sensor(obs_cache):
                return np.concatenate((self.robots[0].ee_force, self.robots[0].ee_torque))

            # hole-related observables
            @sensor(modality=modality)
            def hole_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.hole_body_id])

            @sensor(modality=modality)
            def hole_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.hole_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_hole_pos(obs_cache):
                return obs_cache[f"{pf}eef_pos"] - obs_cache["hole_pos"] if \
                    f"{pf}eef_pos" in obs_cache and "hole_pos" in obs_cache else np.zeros(3)

            sensors = [hole_pos, hole_quat, gripper_to_hole_pos, ft_sensor]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()
            #print(object_placements)

            #print(self.hole.joints)
            #TODO: Figure out how to spawn fixed objects
            if self.hole.joints != []:
                # Loop through all objects and reset their positions
                for obj_pos, obj_quat, obj in object_placements.values():
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the hole.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the hole
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.hole)

    def _check_success(self):
        """
        Check if peg is successfully aligned and placed within the hole

        Returns:
            bool: True if peg is placed in hole correctly
        """
        t, d, cos = self._compute_orientation()

        return d < 0.06 and -0.12 <= t <= 0.14 and cos > 0.95

    def _compute_orientation(self):
        """
        Helper function to return the relative positions between the hole and the peg.
        In particular, the intersection of the line defined by the peg and the plane
        defined by the hole is computed; the parallel distance, perpendicular distance,
        and angle are returned.

        Returns:
            3-tuple:

                - (float): parallel distance
                - (float): perpendicular distance
                - (float): angle
        """
        peg_mat = self.sim.data.body_xmat[self.peg_body_id]
        peg_mat.shape = (3, 3)
        peg_pos = self.sim.data.body_xpos[self.peg_body_id]

        hole_pos = self.sim.data.body_xpos[self.hole_body_id]
        hole_mat = self.sim.data.body_xmat[self.hole_body_id]
        hole_mat.shape = (3, 3)

        v = peg_mat @ np.array([0, 0, 1])
        v = v / np.linalg.norm(v)
        center = hole_pos + hole_mat @ np.array([0.1, 0, 0])

        t = (center - peg_pos) @ v / (np.linalg.norm(v) ** 2)
        d = np.linalg.norm(np.cross(v, peg_pos - center)) / np.linalg.norm(v)

        hole_normal = hole_mat @ np.array([0, 0, 1])
        return (
            t,
            d,
            abs(np.dot(hole_normal, v) / np.linalg.norm(hole_normal) / np.linalg.norm(v)),
        )

    # def _peg_pose_in_hole_frame(self):
    #     """
    #     A helper function that takes in a named data field and returns the pose of that
    #     object in the base frame.

    #     Returns:
    #         np.array: (4,4) matrix corresponding to the pose of the peg in the hole frame
    #     """
    #     # World frame
    #     peg_pos_in_world = self.sim.data.get_body_xpos(self.peg.root_body)
    #     peg_rot_in_world = self.sim.data.get_body_xmat(self.peg.root_body).reshape((3, 3))
    #     peg_pose_in_world = T.make_pose(peg_pos_in_world, peg_rot_in_world)

    #     # World frame
    #     hole_pos_in_world = self.sim.data.get_body_xpos(self.hole.root_body)
    #     hole_rot_in_world = self.sim.data.get_body_xmat(self.hole.root_body).reshape((3, 3))
    #     hole_pose_in_world = T.make_pose(hole_pos_in_world, hole_rot_in_world)
    #     print(hole_pose_in_world)
    #     world_pose_in_hole = T.pose_inv(hole_pose_in_world)

    #     peg_pose_in_hole = T.pose_in_A_to_pose_in_B(
    #         peg_pose_in_world, world_pose_in_hole
    #     )
    #     return peg_pose_in_hole
