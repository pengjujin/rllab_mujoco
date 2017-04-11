from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from custom_env.reacher_mod import ReacherEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

env = normalize(ReacherEnv())

policy = GaussianMLPPolicy(
    env_spec=env.spec,
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
)
algo.train()

observation = env.reset()
for _ in range(5000):
	env.render()
	action, _ = policy.get_action(observation)
	observation, reward, terminal, _ = env.step(action)
	if terminal:
		break