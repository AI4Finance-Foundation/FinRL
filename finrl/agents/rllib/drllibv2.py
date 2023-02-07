# @Author: Astarag Mohapatra
from __future__ import annotations

import ray

assert (
    ray.__version__ > "2.0.0"
), "Please install ray 2.2.0 by doing 'pip install ray[rllib] ray[tune] lz4' , lz4 is for population based tuning"
from pprint import pprint

from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.rllib.algorithms import Algorithm
from ray.tune import register_env

from ray.air import RunConfig, FailureConfig, ScalingConfig
from ray.tune.tune_config import TuneConfig
from ray.air.config import CheckpointConfig

import psutil

psutil_memory_in_bytes = psutil.virtual_memory().total
ray._private.utils.get_system_memory = lambda: psutil_memory_in_bytes
from typing import Dict, Optional, Any, List, Union


class DRLlibv2:
    """
    It instantiates RLlib model with Ray tune functionality
    Params
    -------------------------------------
    trainable:
        Any Trainable class that takes config as parameter
    train_env:
        Training environment instance
    train_env_name: str
        Name of the training environment
    params: dict
        hyperparameters dictionary
    run_name: str
        tune run name
    framework: str
        "torch" or "tf" for tensorflow
    local_dir: str
         to save the results and tensorboard plots
    num_workers: int
        number of workers
    search_alg
        search space for hyperparameters
    concurrent_trials:
         Number of concurrent hyperparameters trial to run
    num_samples: int
         Number of samples of hyperparameters config to run
    scheduler:
        Stopping suboptimal trials
    log_level: str = "WARN",
        Verbosity: "DEBUG"
    num_gpus: Union[float, int] = 0
        GPUs for trial
    num_cpus: Union[float, int] = 2
        CPUs for rollout collection
    dataframe_save: str
        Saving the tune results
    metric: str
        Metric for hyperparameter optimization in Bayesian Methods
    mode: str
        Maximize or Minimize the metric
    max_failures: int
        Number of failures to TuneError
    training_iterations: str
         Number of times session.report() is called
    checkpoint_num_to_keep: int
        Number of checkpoints to keep
    checkpoint_freq: int
        Checkpoint freq wrt training iterations
    reuse_actors:bool
        Reuse actors for tuning

    It has the following methods:
    Methods
    -------------------------------------
        train_tune_model: It takes in the params dictionary and fits in sklearn style to our trainable class
        restore_agent: It restores previously errored or stopped trials or experiments
        infer_results: It returns the results dataframe and trial informations
        get_test_agent: It returns the testing agent for inference

    Example
    ---------------------------------------
    def sample_ppo_params():
        return {
            "entropy_coeff": tune.loguniform(0.00000001, 0.1),
            "lr": tune.loguniform(5e-5, 0.001),
            "sgd_minibatch_size": tune.choice([ 32, 64, 128, 256, 512]),
            "lambda": tune.choice([0.1,0.3,0.5,0.7,0.9,1.0]),
        }
    optuna_search = OptunaSearch(
        metric="episode_reward_mean",
        mode="max")
    drl_agent = DRLlibv2(
        trainable="PPO",
        train_env=env(train_env_config),
        train_env_name="StockTrading_train",
        framework="torch",
        num_workers=1,
        log_level="DEBUG",
        run_name = 'test',
        local_dir = "test",
        params = sample_ppo_params(),
        num_samples = 1,
        num_gpus=1,
        training_iterations=10,
        search_alg = optuna_search,
        checkpoint_freq=5
    )
    #Tune or train the model
    res = drl_agent.train_tune_model()

    #Get the tune results
    results_df, best_result = drl_agent.infer_results()

    #Get the best testing agent
    test_agent = drl_agent.get_test_agent(test_env_instance,'StockTrading_testenv')
    """

    def __init__(
        self,
        trainable: str | Any,
        train_env_name: str,
        params: dict,
        train_env=None,
        run_name: str = "tune_run",
        framework: str = "torch",
        local_dir: str = "tune_results",
        num_workers: int = 1,
        search_alg=None,
        concurrent_trials: int = 0,
        num_samples: int = 0,
        scheduler=None,
        log_level: str = "WARN",
        num_gpus: float | int = 0,
        num_cpus: float | int = 2,
        dataframe_save: str = "tune.csv",
        metric: str = "episode_reward_mean",
        mode: str | list[str] = "max",
        max_failures: int = 0,
        training_iterations: int = 100,
        checkpoint_num_to_keep: None | int = None,
        checkpoint_freq: int = 0,
        reuse_actors: bool = False,
    ):
        if train_env is not None:
            register_env(train_env_name, lambda config: train_env)

        self.params = params
        self.params["framework"] = framework
        self.params["log_level"] = log_level
        self.params["num_gpus"] = num_gpus
        self.params["num_workers"] = num_workers
        self.params["env"] = train_env_name

        self.run_name = run_name
        self.local_dir = local_dir
        self.search_alg = search_alg
        if concurrent_trials != 0:
            self.search_alg = ConcurrencyLimiter(
                self.search_alg, max_concurrent=concurrent_trials
            )
        self.scheduler = scheduler
        self.num_samples = num_samples
        self.trainable = trainable
        if isinstance(self.trainable, str):
            self.trainable.upper()
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.dataframe_save = dataframe_save
        self.metric = metric
        self.mode = mode
        self.max_failures = max_failures
        self.training_iterations = training_iterations
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_num_to_keep = checkpoint_num_to_keep
        self.reuse_actors = reuse_actors

    def train_tune_model(self):
        """
        Tuning and training the model
        Returns the results object
        """
        ray.init(
            num_cpus=self.num_cpus, num_gpus=self.num_gpus, ignore_reinit_error=True
        )

        tuner = tune.Tuner(
            self.trainable,
            param_space=self.params,
            tune_config=TuneConfig(
                search_alg=self.search_alg,
                num_samples=self.num_samples,
                metric=self.metric,
                mode=self.mode,
                reuse_actors=self.reuse_actors,
            ),
            run_config=RunConfig(
                name=self.run_name,
                local_dir=self.local_dir,
                failure_config=FailureConfig(
                    max_failures=self.max_failures, fail_fast=False
                ),
                stop={"training_iteration": self.training_iterations},
                checkpoint_config=CheckpointConfig(
                    num_to_keep=self.checkpoint_num_to_keep,
                    checkpoint_score_attribute=self.metric,
                    checkpoint_score_order=self.mode,
                    checkpoint_frequency=self.checkpoint_freq,
                    checkpoint_at_end=True,
                ),
                verbose=3,
            ),
        )

        self.results = tuner.fit()
        if self.search_alg is not None:
            self.search_alg.save_to_dir(self.local_dir)
        # ray.shutdown()
        return self.results

    def infer_results(self, to_dataframe: str = None, mode: str = "a"):
        """
        Get tune results in a dataframe and best results object
        """
        results_df = self.results.get_dataframe()

        if to_dataframe is None:
            to_dataframe = self.dataframe_save

        results_df.to_csv(to_dataframe, mode=mode)

        best_result = self.results.get_best_result()
        # best_result = self.results.get_best_result()
        # best_metric = best_result.metrics
        # best_checkpoint = best_result.checkpoint
        # best_trial_dir = best_result.log_dir
        # results_df = self.results.get_dataframe()

        return results_df, best_result

    def restore_agent(
        self,
        checkpoint_path: str = "",
        restore_search: bool = False,
        resume_unfinished: bool = True,
        resume_errored: bool = False,
        restart_errored: bool = False,
    ):
        """
        Restore errored or stopped trials
        """
        # if restore_search:
        # self.search_alg = self.search_alg.restore_from_dir(self.local_dir)
        if checkpoint_path == "":
            checkpoint_path = self.results.get_best_result().checkpoint._local_path

        restored_agent = tune.Tuner.restore(
            checkpoint_path,
            restart_errored=restart_errored,
            resume_unfinished=resume_unfinished,
            resume_errored=resume_errored,
        )
        print(restored_agent)
        self.results = restored_agent.fit()

        if self.search_alg is not None:
            self.search_alg.save_to_dir(self.local_dir)
        return self.results

    def get_test_agent(self, test_env_name: str, test_env=None, checkpoint=None):
        """
        Get test agent
        """
        if test_env is not None:
            register_env(test_env_name, lambda config: test_env)

        if checkpoint is None:
            checkpoint = self.results.get_best_result().checkpoint

        testing_agent = Algorithm.from_checkpoint(checkpoint)
        # testing_agent.config['env'] = test_env_name

        return testing_agent
