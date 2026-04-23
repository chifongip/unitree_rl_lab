import argparse
from isaaclab.app import AppLauncher

# 1. Setup the AppLauncher (required before importing other Isaac Lab modules)
parser = argparse.ArgumentParser(description="Load a USD and check joint order.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. Imports that depend on the simulation app running
import torch
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg, SimulationContext, UsdFileCfg
from isaaclab.utils import configclass

from isaaclab.actuators import ImplicitActuatorCfg

@configclass
class RobotSceneCfg:
    """Design the scene with the robot."""
    sim: SimulationCfg = SimulationCfg(device="cuda", dt=0.01)
    
    robot: ArticulationCfg = ArticulationCfg(
        spawn=UsdFileCfg(
            usd_path="unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd",
        ),
        # FIX 1: Define where the robot is spawned in the USD Stage tree
        prim_path="/World/Robot",
        # FIX 2: Define the actuator models (standard PD control)
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],  # Applies this to all joints
                stiffness=40.0,
                damping=2.0,
            ),
        },
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )

def main():
    # 3. Setup the simulation context
    scene_cfg = RobotSceneCfg()
    sim = SimulationContext(scene_cfg.sim)
    
    # 4. Instantiate the robot in the scene
    robot = Articulation(cfg=scene_cfg.robot)
    
    # 5. Play the simulation to initialize physics buffers
    sim.reset()
    
    print("-" * 40)
    print("ROBOT JOINT INFORMATION")
    print("-" * 40)
    
    # Access joint names from the data object
    joint_names = robot.data.joint_names
    print(f"Total Joints: {len(joint_names)}")
    
    # Print the ordered list
    for i, name in enumerate(joint_names):
        print(f"Index {i:2d}: {name}")
    
    print("-" * 40)

    # 6. Keep the app open for a few steps to visualize
    while simulation_app.is_running():
        # Step physics
        sim.step()
        # Update robot data buffers
        robot.update(sim.get_physics_dt())

if __name__ == "__main__":
    main()
    simulation_app.close()