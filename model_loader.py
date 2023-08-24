import wandb
from agents.rl_birdview.rl_birdview_agent import RlBirdviewAgent
import ipdb
import gym
import json
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os.path
import sys
class ModelLoader:
    def __init__(self, path_to_conf_file='config_agent.yaml'):
        self.path_to_conf_file = path_to_conf_file
        self.load_config()
        self.load_wandb_model()

    def load_config(self):
        self.cfg = OmegaConf.load(self.path_to_conf_file)

    def load_wandb_model(self):
        # Load the model using your agent class
        self.agent = RlBirdviewAgent(path_to_conf_file='config_agent.yaml')
    
    # def run_environment(self, env_setup, max_step=None):
    #     # Create the environment
    #     env = gym.make(env_setup['env_id'], obs_configs=obs_configs, reward_configs=reward_configs,
    #                    terminal_configs=terminal_configs, host=cfg.host, port=cfg.port,
    #                    seed=cfg.seed, no_rendering=cfg.no_rendering, **env_setup['env_configs'])

    #     # Run the environment using the loaded agent
    #     list_render, ep_stat_dict, ep_event_dict, timestamp = run_single(
    #         run_name, env, {'actor_id': self.agent}, agents_log_dir, False, max_step)

    #     # Close the environment
    #     env.close()

    #     return list_render, ep_stat_dict, ep_event_dict, timestamp

    def get_action(self, obs, timestamp):
        # # Create an instance of ModelLoader to load the model
        # model_loader = ModelLoader()
        # Call the run_step function using the loaded agent
        control = self.agent.run_step(obs, timestamp)
        
        return control

