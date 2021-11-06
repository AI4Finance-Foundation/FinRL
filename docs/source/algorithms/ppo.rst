:github_url: https://github.com/AI4Finance-LLC/FinRL-Library

PPO
================

We explore and use PPO as a component in the ensemble method. PPO is introduced to control the policy gradient update and ensure that the new policy will not be too different from the older one. PPO tries to simplify the objective of Trust Region Policy Optimization (TRPO) by introducing a clipping term to the objective function.

The objective function of PPO takes the minimum of the clipped and normal objective. PPO discourages large policy change move outside of the clipped interval. Therefore, PPO improves the stability of the policy networks training by restricting the policy update at each training step. We select PPO for stock trading because it is stable, fast, and simpler to implement and tune.


Example
-------------------

.. code-block:: python
    :linenos:
    
    def train_PPO(self, model_name, model_params = config.PPO_PARAMS):
        """PPO model"""
        from stable_baselines import PPO2
        env_train = self.env

        start = time.time()
        model = PPO2('MlpPolicy', env_train,
                     n_steps = model_params['n_steps'],
                     ent_coef = model_params['ent_coef'],
                     learning_rate = model_params['learning_rate'],
                     nminibatches = model_params['nminibatches'],
                     verbose = model_params['verbose'],
                     tensorboard_log = f"{config.TENSORBOARD_LOG_DIR}/{model_name}"
                     )
        model.learn(total_timesteps=model_params['timesteps'], tb_log_name = "PPO_run")
        end = time.time()

        model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
        print('Training time (PPO): ', (end-start)/60,' minutes')
        return model
        
Actor-Critic
----------------------

.. function:: finrl.model.models.DRLAgent.train_PPO(self, model_name, model_params = config.PPO_PARAMS)

    .. function:: PPO2('MlpPolicy', env_train, n_steps = model_params['n_steps'], ent_coef = model_params['ent_coef'], learning_rate = model_params['learning_rate'], nminibatches = model_params['nminibatches'], verbose = model_params['verbose'], tensorboard_log = f"{config.TENSORBOARD_LOG_DIR}/{model_name}")
                     
        :param MlpPolicy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, …)
        :param env_train: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
        :param n_steps: (int) The number of steps to run for each environment per update (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
        :param ent_coef:  (float) Entropy coefficient for the loss calculation
        :param learning_rate:  (float or callable) The learning rate, it can be a function
        :param nminibatches: (int) Number of training minibatches per update. For recurrent policies, the number of environments run in parallel should be a multiple of nminibatches.
        :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
        :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
        