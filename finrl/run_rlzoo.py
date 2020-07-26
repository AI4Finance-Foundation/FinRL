from rlzoo.common.env_wrappers import *
from rlzoo.common.utils import *
from rlzoo.algorithms import *

# EnvName = 'PongNoFrameskip-v4'
# EnvType = 'atari'

# EnvName = 'CartPole-v0'
EnvName = 'Pendulum-v0'
EnvType = 'classic_control'

# EnvName = 'BipedalWalker-v2'
# EnvType = 'box2d'

# EnvName = 'Ant-v2'
# EnvType = 'mujoco'

# EnvName = 'FetchPush-v1'
# EnvType = 'robotics' 

# EnvName = 'FishSwim-v0'
# EnvType = 'dm_control'

# EnvName = 'ReachTarget'
# EnvType = 'rlbench'
# env = build_env(EnvName, EnvType, state_type='vision')

AlgName = 'SAC'
env = build_env(EnvName, EnvType)
alg_params, learn_params = call_default_params(env, EnvType, AlgName)
alg = eval(AlgName+'(**alg_params)')
alg.learn(env=env, mode='train', render=False, **learn_params)
alg.learn(env=env, mode='test', render=True, **learn_params)

# AlgName = 'DPPO'
# number_workers = 2  # need to specify number of parallel workers in parallel algorithms like A3C and DPPO
# env = build_env(EnvName, EnvType, nenv=number_workers)
# alg_params, learn_params = call_default_params(env, EnvType, AlgName)
# alg_params['method'] = 'clip'    # specify 'clip' or 'penalty' method for different version of PPO and DPPO
# alg = eval(AlgName+'(**alg_params)')
# alg.learn(env=env,  mode='train', render=False, **learn_params)
# alg.learn(env=env,  mode='test', render=True, **learn_params)

# AlgName = 'PPO'
# env = build_env(EnvName, EnvType)
# alg_params, learn_params = call_default_params(env, EnvType, AlgName)
# alg_params['method'] = 'clip'    # specify 'clip' or 'penalty' method for different version of PPO and DPPO
# alg = eval(AlgName+'(**alg_params)')
# alg.learn(env=env,  mode='train', render=False, **learn_params)
# alg.learn(env=env,  mode='test', render=True, **learn_params)

# AlgName = 'A3C'
# number_workers = 2  # need to specify number of parallel workers
# env = build_env(EnvName, EnvType, nenv=number_workers)
# alg_params, learn_params = call_default_params(env, EnvType, 'A3C')
# alg = eval(AlgName+'(**alg_params)')
# alg.learn(env=env,  mode='train', render=False, **learn_params)
# alg.learn(env=env,  mode='test', render=True, **learn_params)

env.close()
