from elegantrl.agent import *
from elegantrl.run import *
import torch 
from neo_finrl.data_processor import DataProcessor

def test(start_date, end_date, ticker_list, data_source, time_interval, 
         technical_indicator_list, drl_lib, env, agent, if_vix = True,
         **kwargs):
    #fetch data
    DP = DataProcessor(data_source, **kwargs)
    data = DP.download_data(ticker_list, start_date, end_date, time_interval)
    data = DP.clean_data(data)
    data = DP.add_technical_indicator(data, technical_indicator_list)
    if if_vix:
        data = DP.add_vix(data)
    price_array, tech_array, risk_array = DP.df_to_array(data, if_vix)
    
    env_config = {'price_array':price_array,
            'tech_array':tech_array,
            'risk_array':risk_array,
            'if_train':False}
    env_instance = env(config=env_config)
    
    net_dimension = kwargs.get('net_dimension', 2**7)
    cwd = kwargs.get('cwd','./'+str(agent))

    #test on elegantrl
    if drl_lib == 'elegantrl':
        
        #select agent
        if agent == 'ppo':
            args = Arguments(if_on_policy=True)
            args.agent = AgentPPO()
            args.env = env_instance
            args.agent.if_use_cri_target = True
        else:
            raise ValueError('Invalid agent input or the agent input is not \
                             supported yet.')
        
        #load agent
        try:
            state_dim = env_instance.state_dim
            action_dim = env_instance.action_dim
    
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
        state = env_instance.reset()
        episode_returns = list()  # the cumulative_return / initial_account
        with _torch.no_grad():
            for i in range(env_instance.max_step):
                s_tensor = _torch.as_tensor((state,), device=device)
                a_tensor = act(s_tensor)  # action_tanh = act.forward()
                action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
                state, reward, done, _ = env_instance.step(action)

                total_asset = env_instance.amount + (env_instance.price_ary[env_instance.day] * env_instance.stocks).sum()
                episode_return = total_asset / env_instance.initial_total_asset
                episode_returns.append(episode_return)
                if done:
                    break
        print('Test Finished!')
        #return episode returns on testing data
        return episode_returns
    
    #test using rllib
    elif drl_lib == 'rllib':
        #load agent
        from ray.rllib.agents import ppo
        from ray.rllib.agents.ppo.ppo import PPOTrainer
        config = ppo.DEFAULT_CONFIG.copy()
        config['env'] = env
        config["log_level"] = "WARN"
        config['env_config'] = {'price_array':price_array,
                                'tech_array':tech_array,
                                'risk_array':risk_array,
                                'if_train':False}
        
        trainer = PPOTrainer(env=env, config=config)
        try:
            trainer.restore(cwd)
            print("Restoring from checkpoint path", cwd)
        except:
            raise ValueError('Fail to load agent!')
        
        #test on the testing env
        state = env_instance.reset()
        episode_returns = list()  # the cumulative_return / initial_account
        done = False
        while not done:
            action = trainer.compute_single_action(state)
            state, reward, done, _ = env_instance.step(action)
    
            total_asset = env_instance.amount + (env_instance.price_ary[env_instance.day] * env_instance.stocks).sum()
            episode_return = total_asset / env_instance.initial_total_asset
            episode_returns.append(episode_return)
        print('episode return: ' + str(episode_return))
        print('Test Finished!')   
        return episode_returns

          #test using stable baselines3
    elif drl_lib == 'stable_baselines3':
        from stable_baselines3 import PPO

        try:
            #load agent
            model = PPO.load(cwd)
            print("Successfully load model", cwd)
        except:
            raise ValueError('Fail to load agent!')
        
        #test on the testing env
        state = env_instance.reset()
        episode_returns = list()  # the cumulative_return / initial_account
        done = False
        while not done:
            action = model.predict(state)[0]
            state, reward, done, _ = env_instance.step(action)
    
            total_asset = env_instance.amount + (env_instance.price_ary[env_instance.day] * env_instance.stocks).sum()
            episode_return = total_asset / env_instance.initial_total_asset
            episode_returns.append(episode_return)
        
        print('episode_return', episode_return)
        print('Test Finished!')   
        return episode_returns
    
    else:
        raise ValueError('DRL library input is NOT supported yet. Please check.')
            
if __name__ == '__main__':    
    from neo_finrl.config import FAANG_TICKER
    from neo_finrl.config import TECHNICAL_INDICATORS_LIST
    from neo_finrl.config import TEST_START_DATE
    from neo_finrl.config import TEST_END_DATE
    
    #construct environment
    from neo_finrl.env_stock_trading.env_stock_trading import StockTradingEnv
    env = StockTradingEnv
    
    #demo for elegantrl
    test(start_date = TEST_START_DATE, end_date = TEST_END_DATE,
         ticker_list = FAANG_TICKER, data_source = 'yahoofinance',
         time_interval= '1D', technical_indicator_list= TECHNICAL_INDICATORS_LIST,
         drl_lib='elegantrl', env=env, agent='ppo', 
         cwd='./test_ppo', net_dimension = 2 ** 9)
    
    #demo for rllib 
    test(start_date = TEST_START_DATE, end_date = TEST_END_DATE,
         ticker_list = FAANG_TICKER, data_source = 'yahoofinance',
         time_interval= '1D', technical_indicator_list= TECHNICAL_INDICATORS_LIST,
         drl_lib='rllib', env=env, agent='ppo', 
         cwd='./test_ppo/checkpoint_000010/checkpoint-10')

    #demo for stable baselines3 
    test(start_date = TEST_START_DATE, end_date = TEST_END_DATE,
         ticker_list = FAANG_TICKER, data_source = 'yahoofinance',
         time_interval= '1D', technical_indicator_list= TECHNICAL_INDICATORS_LIST, 
         drl_lib='stable_baselines3', env=env, agent='ppo', 
         cwd='./test_ppo.zip')

