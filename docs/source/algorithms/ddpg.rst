:github_url: https://github.com/AI4Finance-LLC/FinRL-Library

DDPG
=======================

DDPG is an actor-critic based algorithm that combines both Q-learning and policy gradient, and uses neural networks as function approximators. In contrast with DQN that learns indirectly through Q-values and suffers the curse of dimensionality problem, DDPG learns directly from the observations through policy gradient. It is proposed to deterministically map states to actions to better fit the continuous action space environment.

.. image:: ../image/ddpg.png

Example
-------------------

.. code-block:: python
    :linenos:

    def train_DDPG(self, model_name, model_params = config.DDPG_PARAMS):
        """DDPG model"""
        from stable_baselines import DDPG
        from stable_baselines.ddpg.policies import DDPGPolicy
        from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise


        env_train = self.env

        n_actions = env_train.action_space.shape[-1]
        param_noise = None
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5)*np.ones(n_actions))

        start = time.time()
        model = DDPG('MlpPolicy', 
                    env_train,
                    batch_size=model_params['batch_size'],
                    buffer_size=model_params['buffer_size'],
                    param_noise=param_noise,
                    action_noise=action_noise,
                    verbose=model_params['verbose'],
                    tensorboard_log = f"{config.TENSORBOARD_LOG_DIR}/{model_name}"
                    )
        model.learn(total_timesteps=model_params['timesteps'], tb_log_name = "DDPG_run")
        end = time.time()

        model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
        print('Training time (DDPG): ', (end-start)/60,' minutes')
        return model

Actor-Critic
----------------------

.. function:: finrl.model.models.DRLAgent.train_DDPG(self, model_name, model_params = config.DDPG_PARAMS)

    .. function:: DDPG('MlpPolicy', env_train, batch_size=model_params['batch_size'], buffer_size=model_params['buffer_size'], param_noise=param_noise, action_noise=action_noise, verbose=model_params['verbose'], tensorboard_log = f"{config.TENSORBOARD_LOG_DIR}/{model_name}" )
        
        :param MlpPolicy: (DDPGPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, …)
        :param env_train: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
        :param batch_size: (int) the size of the batch for learning the policy
        :param buffer_size: (int) the max number of transitions to store, size of the replay buffer
        :param param_noise:  (AdaptiveParamNoiseSpec) the parameter noise type (can be None)
        :param action_noise: (ActionNoise) the action noise type (can be None)
        :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
        :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
        
        
