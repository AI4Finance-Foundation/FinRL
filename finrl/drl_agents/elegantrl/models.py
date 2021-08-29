# common library
import torch 
# RL models from elegantrl
from elegantrl.agent import AgentPPO
from elegantrl.agent import AgentTD3
from elegantrl.agent import AgentDDPG
from elegantrl.agent import AgentSAC

from elegantrl.run import *

MODELS = {"ddpg": AgentDDPG, "td3": AgentTD3, "sac": AgentSAC, "ppo": AgentPPO}

'''MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}'''


class DRLAgent:
    """Provides implementations for DRL algorithms
    Attributes
    ----------
        env: gym environment class
            user-defined class
    Methods
    -------
        train_PPO()
            the implementation for PPO algorithm
        train_A2C()
            the implementation for A2C algorithm
        train_DDPG()
            the implementation for DDPG algorithm
        train_TD3()
            the implementation for TD3 algorithm
        train_SAC()
            the implementation for SAC algorithm
        DRL_prediction()
            make a prediction in a test dataset and get results
    """
    def __init__(self, 
                 env, 
                 price_array,
                 tech_array,
                 turbulence_array):
        self.env = env
        self.price_array=price_array
        self.tech_array=tech_array
        self.turbulence_array=turbulence_array
        
    def get_model(
        self,
        model_name,
        model_kwargs
    ):
        
        model = Arguments(if_on_policy=True)
        
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        
        if model_kwargs is not None:
            try:
                model.learning_rate = model_kwargs['learning_rate']
                model.batch_size = model_kwargs['batch_size']
                model.gamma = model_kwargs['gamma']
                model.seed = model_kwargs['seed']
                model.net_dimension = model_kwargs['net_dimension']
            except:
                raise ValueError("Fail to read arguments, please check 'model_kwargs' input.")
        
        model.agent = MODELS[model_name]()
        env_config = {'price_array':self.price_array,
        'tech_array':self.tech_array,
        'turbulence_array':self.turbulence_array,
        'if_train':True}
        model.env = self.env(config=env_config)
        return model

    def train_model(self, model, cwd, total_timesteps=5000):  
        model.cwd = cwd
        model.break_step = total_timesteps      
        train_and_evaluate(model)
    
    @staticmethod
    def DRL_prediction(model_name, cwd, net_dimension, environment):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        model = MODELS[model_name]
        args = Arguments(agent=model(), env=environment, if_on_policy=True)
        
        #load agent
        try:
            state_dim = environment.state_dim
            action_dim = environment.action_dim
    
            agent = args.agent
            net_dim = net_dimension
    
            agent.init(net_dim, state_dim, action_dim)
            agent.save_or_load_agent(cwd=cwd, if_save=False)
            act = agent.act
            device = agent.device
    
        except:
            raise ValueError('Fail to load agent!')
        
        #test on the testing env
        _torch = torch
        state = environment.reset()
        episode_total_assets = list()
        with _torch.no_grad():
            for i in range(environment.max_step):
                s_tensor = _torch.as_tensor((state,), device=device)
                a_tensor = act(s_tensor)  # action_tanh = act.forward()
                action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
                state, reward, done, _ = environment.step(action)

                total_asset = environment.amount + (environment.price_ary[environment.day] * environment.stocks).sum()
                episode_total_assets.append(total_asset)
                if done:
                    break
        print('Test Finished!')
        #return episode total_assets on testing data
        return episode_total_assets