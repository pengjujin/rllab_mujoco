from reacher_mod import ReacherEnv
from gym.envs.registration import register

register(
	id='Reacher_mod-v0',
	entry_point='env.reacher_mod:ReacherEnv',
	max_episode_steps=1000,
	# reward_threshold=-3.75,
)