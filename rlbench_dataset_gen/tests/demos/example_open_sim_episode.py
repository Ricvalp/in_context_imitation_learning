# example_open_sim_episode.py
import time

from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.utils import task_file_to_task_class

TASK = "open_wine_bottle"
VARIATION = 0

env = Environment(
    action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
    obs_config=ObservationConfig(),
    headless=False,
)
env.launch()

task_cls = task_file_to_task_class(f"{TASK}.py")
task_env = env.get_task(task_cls)
task_env.set_variation(VARIATION)
task_env.reset()

print("CoppeliaSim is running. Ctrl+C here when done.")
try:
    while True:
        env._scene.pyrep.step()  # keep UI responsive
        time.sleep(0.01)
except KeyboardInterrupt:
    env.shutdown()
