import wandb
from omegaconf import OmegaConf
from agents.rl_birdview.rl_birdview_agent import RlBirdviewAgent
import ipdb
import os

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

# Create an instance of ModelLoader to load the model
model_loader = ModelLoader()


# # Assuming you have a model_loader instance
# loaded_agent = model_loader.agent

# # Use the loaded agent as needed
# observation = ...  # Your input observation
# action = loaded_agent.predict(observation)

