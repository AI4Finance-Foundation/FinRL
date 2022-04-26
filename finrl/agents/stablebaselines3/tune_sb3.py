import datetime
import sys
from pprint import pprint

import joblib
import optuna
import pandas as pd
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl import config
from main import check_and_make_directories
import hyperparams_opt as hpt
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from finrl.plot import backtest_plot, backtest_stats, get_baseline, get_daily_return
from typing import Union, Tuple

class LoggingCallback:
    def __init__(self, threshold:int, trial_number:int, patience:int):
        """
      threshold:int tolerance for increase in sharpe ratio
      trial_number: int Prune after minimum number of trials
      patience: int patience for the threshold
      """
        self.threshold = threshold
        self.trial_number = trial_number
        self.patience = patience
        self.cb_list = []  # Trials list for which threshold is reached

    def __call__(self, study: optuna.study, frozen_trial: optuna.Trial):
        # Setting the best value in the current trial
        study.set_user_attr("previous_best_value", study.best_value)

        # Checking if the minimum number of trials have pass
        if frozen_trial.number > self.trial_number:
            previous_best_value = study.user_attrs.get("previous_best_value", None)
            # Checking if the previous and current objective values have the same sign
            if previous_best_value * study.best_value >= 0:
                # Checking for the threshold condition
                if abs(previous_best_value - study.best_value) < self.threshold:
                    self.cb_list.append(frozen_trial.number)
                    # If threshold is achieved for the patience amount of time
                    if len(self.cb_list) > self.patience:
                        print("The study stops now...")
                        print(
                            "With number",
                            frozen_trial.number,
                            "and value ",
                            frozen_trial.value,
                        )
                        print(
                            "The previous and current best values are {} and {} respectively".format(
                                previous_best_value, study.best_value
                            )
                        )
                        study.stop()


class TuneSB3Optuna:
    """
  Hyperparameter tuning of SB3 agents using Optuna

  Attributes
  ---------- 
    env_train: Training environment for SB3
    model_name: str
    env_trade: testing environment
    logging_callback: callback for tuning
    total_timesteps: int
    n_trials: number of hyperparameter configurations
  
  Note:
    The default sampler and pruner are used are
    Tree Parzen Estimator and Hyperband Scheduler
    respectively.
  """

    def __init__(
        self,
        env_train,
        model_name: str,
        env_trade,
        logging_callback,
        total_timesteps: int = 50000,
        n_trials: int = 30,
    ):

        self.env_train = env_train
        self.agent = DRLAgent(env=env_train)
        self.model_name = model_name
        self.env_trade = env_trade
        self.total_timesteps = total_timesteps
        self.n_trials = n_trials
        self.logging_callback = logging_callback
        self.MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}

        check_and_make_directories(
            [
                config.DATA_SAVE_DIR,
                config.TRAINED_MODEL_DIR,
                config.TENSORBOARD_LOG_DIR,
                config.RESULTS_DIR,
            ]
        )

    def default_sample_hyperparameters(self, trial: optuna.Trial):
        if self.model_name == "a2c":
            return hpt.sample_a2c_params(trial)
        elif self.model_name == "ddpg":
            return hpt.sample_ddpg_params(trial)
        elif self.model_name == "td3":
            return hpt.sample_td3_params(trial)
        elif self.model_name == "sac":
            return hpt.sample_sac_params(trial)
        elif self.model_name == "ppo":
            return hpt.sample_ppo_params(trial)

    def calculate_sharpe(self, df: pd.DataFrame):
        df["daily_return"] = df["account_value"].pct_change(1)
        if df["daily_return"].std() != 0:
            sharpe = (252 ** 0.5) * df["daily_return"].mean() / df["daily_return"].std()
            return sharpe
        else:
            return 0

    def objective(self, trial: optuna.Trial):
        hyperparameters = self.default_sample_hyperparameters(trial)
        policy_kwargs = hyperparameters["policy_kwargs"]
        del hyperparameters["policy_kwargs"]
        model = self.agent.get_model(
            self.model_name, policy_kwargs=policy_kwargs, model_kwargs=hyperparameters
        )
        trained_model = self.agent.train_model(
            model=model,
            tb_log_name=self.model_name,
            total_timesteps=self.total_timesteps,
        )
        trained_model.save(
            f"./{config.TRAINED_MODEL_DIR}/{self.model_name}_{trial.number}.pth"
        )
        df_account_value, _ = DRLAgent.DRL_prediction(
            model=trained_model, environment=self.env_trade
        )
        sharpe = self.calculate_sharpe(df_account_value)

        return sharpe

    def run_optuna(self):
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
            study_name=f"{self.model_name}_study",
            direction="maximize",
            sampler=sampler,
            pruner=optuna.pruners.HyperbandPruner(),
        )

        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            catch=(ValueError,),
            callbacks=[self.logging_callback],
        )

        joblib.dump(study, f"{self.model_name}_study.pkl")
        return study

    def backtest(self, final_study: optuna.Study) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        print("Hyperparameters after tuning", final_study.best_params)
        print("Best Trial", final_study.best_trial)

        tuned_model = self.MODELS[self.model_name].load(
            f"./{config.TRAINED_MODEL_DIR}/{self.model_name}_{final_study.best_trial.number}.pth",
            env=self.env_train,
        )

        df_account_value_tuned, df_actions_tuned = DRLAgent.DRL_prediction(
            model=tuned_model, environment=self.env_trade
        )

        print("==============Get Backtest Results===========")
        now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")

        perf_stats_all_tuned = backtest_stats(account_value=df_account_value_tuned)
        perf_stats_all_tuned = pd.DataFrame(perf_stats_all_tuned)
        perf_stats_all_tuned.to_csv(
            "./" + config.RESULTS_DIR + "/perf_stats_all_tuned_" + now + ".csv"
        )

        return df_account_value_tuned, df_actions_tuned, perf_stats_all_tuned
