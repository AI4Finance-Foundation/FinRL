:github_url: https://github.com/AI4Finance-LLC/FinRL-Library

SAC
===================

Example
-------------------

.. code-block:: python
    :linenos:

    def train_SAC(self, model_name, model_params = config.SAC_PARAMS):
        """TD3 model"""
        from stable_baselines import SAC

        env_train = self.env

        start = time.time()
        model = SAC('MlpPolicy', env_train,
                    batch_size=model_params['batch_size'],
                    buffer_size=model_params['buffer_size'],
                    learning_rate = model_params['learning_rate'],
                    learning_starts=model_params['learning_starts'],
                    ent_coef=model_params['ent_coef'],
                    verbose=model_params['verbose'],
                    tensorboard_log = f"{config.TENSORBOARD_LOG_DIR}/{model_name}"
                    )
        model.learn(total_timesteps=model_params['timesteps'], tb_log_name = "SAC_run")
        end = time.time()

        model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
        print('Training time (SAC): ', (end-start)/60,' minutes')
        return model


Actor-Critic
----------------------

.. function:: finrl.model.models.DRLAgent.train_SAC(self, model_name, model_params = config.SAC_PARAMS)

    .. function:: SAC('MlpPolicy', env_train, batch_size=model_params['batch_size'], buffer_size=model_params['buffer_size'], learning_rate = model_params['learning_rate'], learning_starts=model_params['learning_starts'], ent_coef=model_params['ent_coef'], verbose=model_params['verbose'], tensorboard_log = f"{config.TENSORBOARD_LOG_DIR}/{model_name}")
                     
        :param MlpPolicy: (SACPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, …)
        :param env_train: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
        :param batch_size: (int) Minibatch size for each gradient update
        :param buffer_size: (int) size of the replay buffer
        :param learning_rate: (float or callable) learning rate for adam optimizer, the same learning rate will be used for all networks (Q-Values, Actor and Value function) it can be a function of the current progress (from 1 to 0)
        :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
        :param ent_coef:  (str or float) Entropy regularization coefficient. (Equivalent to inverse of reward scale in the original SAC paper.) Controlling exploration/exploitation trade-off. Set it to ‘auto’ to learn it automatically (and ‘auto_0.1’ for using 0.1 as initial value)
        :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
        :param tensorboard_log:  (str) the log location for tensorboard (if None, no logging)
        