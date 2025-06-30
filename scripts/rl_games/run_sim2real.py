import pprint
import time
import torch
import yaml

# from hw_wrappers.unitree_h1 import UnitreeH1  # noqa
from inference_env.deplotment_player import DeploymentPlayer  # noqa 
# from sim2real.deployment_player import DeploymentPlayer

from sim2real.inference_env.Agile_env_cfg_real import AgileDroneEnvCfgReal
from sim2real.inference_env.utils import get_player_args


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")