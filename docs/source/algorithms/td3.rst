:github_url: https://github.com/AI4Finance-LLC/FinRL-Library

TD3
================


Example
-------------------

.. code-block:: python
    :linenos:

    def train_TD3(self, model_name, model_params = config.TD3_PARAMS):
        """TD3 model"""
        from stable_baselines import TD3
        from stable_baselines.common.noise import NormalActionNoise

        env_train = self.env

        n_actions = env_train.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions))

        start = time.time()
        model = TD3('MlpPolicy', env_train,
                    batch_size=model_params['batch_size'],
                    buffer_size=model_params['buffer_size'],
                    learning_rate = model_params['learning_rate'],
                    action_noise = action_noise,
                    verbose=model_params['verbose'],
                    tensorboard_log = f"{config.TENSORBOARD_LOG_DIR}/{model_name}"
                    )
        model.learn(total_timesteps=model_params['timesteps'], tb_log_name = "TD3_run")
        end = time.time()

        model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
        print('Training time (DDPG): ', (end-start)/60,' minutes')
        return model

Actor-Critic
----------------------

.. function:: finrl.model.models.DRLAgent.train_TD3(self, model_name, model_params = config.TD3_PARAMS)

    .. function:: TD3('MlpPolicy', env_train, batch_size=model_params['batch_size'], buffer_size=model_params['buffer_size'], learning_rate = model_params['learning_rate'], action_noise = action_noise, verbose=model_params['verbose'], tensorboard_log = f"{config.TENSORBOARD_LOG_DIR}/{model_name}")
                    
        :param MlpPolicy: (TD3Policy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, …)
        :param env_train: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
        :param batch_size: (int) Minibatch size for each gradient update
        :param buffer_size: (int) size of the replay buffer
        :param learning_rate: (float or callable) learning rate for adam optimizer, the same learning rate will be used for all networks (Q-Values and Actor networks) it can be a function of the current progress (from 1 to 0)
        :param action_noise: (ActionNoise) the action noise type. Cf DDPG for the different action noise type.
        :param verbose:  (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
        :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
        