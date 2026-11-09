"""Isaac Lab v2.1.1 adapters; import SDK classes only after AppLauncher."""
import math
from functools import partial
from .lidar import horizontal_pattern
from .contact import make_contact_configs


def make_articulation_cfg(prim_path, start, start_yaw, robot_usd=None):
    from isaaclab.assets import ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    import isaaclab.sim as sim_utils
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=robot_usd or f'{ISAAC_NUCLEUS_DIR}/Robots/Jetbot/jetbot.usd',
            activate_contact_sensors=True),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(*start, 0.05), rot=(math.cos(start_yaw/2), 0., 0., math.sin(start_yaw/2))),
        actuators={'wheels': ImplicitActuatorCfg(joint_names_expr=['.*'], stiffness=None, damping=None)})


def make_sensor_configs(cfg, chassis_path, body_paths, scan_mesh_path, obstacle_filters,
                        mount_offset=(0., 0., .20), debug_vis=False):
    from isaaclab.sensors import RayCasterCfg
    from isaaclab.sensors.ray_caster.patterns import PatternBaseCfg
    lidar = RayCasterCfg(
        prim_path=chassis_path, update_period=cfg.control_dt,
        offset=RayCasterCfg.OffsetCfg(pos=mount_offset), ray_alignment='base',
        mesh_prim_paths=[scan_mesh_path],
        pattern_cfg=PatternBaseCfg(func=partial(horizontal_pattern, navigation_cfg=cfg)),
        max_distance=cfg.lidar_range, drift_range=(0., 0.),
        ray_cast_drift_range={'x': (0., 0.), 'y': (0., 0.), 'z': (0., 0.)}, debug_vis=debug_vis)
    return lidar, make_contact_configs(cfg, body_paths, chassis_path, obstacle_filters, debug_vis)
