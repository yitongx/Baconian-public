# from baconian.benchmark.ppo_benchmark.halfCheetah import half_cheetah_task_fn
# from baconian.benchmark.ppo_benchmark.pendulum import pendulum_task_fn
# from baconian.benchmark.ppo_benchmark.reacher import reacher_task_fn
# from baconian.benchmark.ppo_benchmark.swimmer import swimmer_task_fn
# from baconian.benchmark.ppo_benchmark.hopper import hopper_task_fn
# from baconian.benchmark.ppo_benchmark.inverted_pendulum import inverted_pendulum_task_fn
# from baconian.benchmark.ppo_benchmark.halfCheetah_pybullet import \half_cheetah_task_fn as half_cheetah_bullet_env_task_fn
from baconian.benchmark.ppo_benchmark.mujoco_bullet_env import half_cheetah_bullet_env_task_fn, \
    inverted_pendulum_bullet_env_task_fn, inverted_double_pendulum_bullet_env_task_fn, pendulum_env_task_fn
