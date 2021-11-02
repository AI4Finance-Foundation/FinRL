:github_url: https://github.com/AI4Finance-LLC/FinRL-Library

A2C
================

A2C is a typical actor-critic algorithm that is introduced to improve the policy gradient updates. A2C utilizes an advantage function to reduce the variance of the policy gradient. Instead of only estimates the value function, the critic network estimates the advantage function. Thus, the evaluation of an action not only depends on how good the action is, but also considers how much better it can be. So that it reduces the high variance of the policy networks and makes the model more robust.

A2C uses copies of the same agent working in parallel to update gradients with different data samples. Each agent works independently to interact with the same environment. After all of the parallel agents finish calculating their gradients, A2C uses a coordinator to pass the average gradients over all the agents to a global network. So that the global network can update the actor and the critic network. The presence of a global network increases the diversity of training data. The synchronized gradient update is more cost-effective, faster and works better with large batch sizes. A2C is a great model for stock trading because of its stability.

.. image:: ../image/a2c.png

Example
-------------------

.. code-block:: python
    :linenos:

    def train_A2C(self, model_name, model_params = config.A2C_PARAMS):
            """A2C model"""
            from stable_baselines import A2C
            env_train = self.env
            start = time.time()
            model = A2C('MlpPolicy', env_train, 
                        n_steps = model_params['n_steps'],
                        ent_coef = model_params['ent_coef'],
                        learning_rate = model_params['learning_rate'],
                        verbose = model_params['verbose'],
                        tensorboard_log = f"{config.TENSORBOARD_LOG_DIR}/{model_name}"
                        )
            model.learn(total_timesteps=model_params['timesteps'], tb_log_name = "A2C_run")
            end = time.time()
    
            model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
            print('Training time (A2C): ', (end-start)/60,' minutes')
            return model


Actor-Critic
----------------------

.. function:: finrl.model.models.DRLAgent.train_A2C(self, model_name, model_params = config.A2C_PARAMS)

    .. function:: A2C('MlpPolicy', env_train, n_steps = model_params['n_steps'], ent_coef = model_params['ent_coef'], learning_rate = model_params['learning_rate'], verbose = model_params['verbose'], tensorboard_log = f"{config.TENSORBOARD_LOG_DIR}/{model_name}")
        
        
        :param MlpPolicy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, …)
        :param env_train: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
        :param n_steps: (int) The number of steps to run for each environment per update (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
        :param ent_coef: (float) Entropy coefficient for the loss calculation
        :param learning_rate: (float) The learning rate
        :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
        :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
        
        

