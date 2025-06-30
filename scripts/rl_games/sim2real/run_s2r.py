import pprint
import time
import torch
import yaml

# from hw_wrappers.unitree_h1 import UnitreeH1  # noqa
from inference_env.deplotment_player import DeploymentPlayer  # noqa 
# from sim2real.deployment_player import DeploymentPlayer

from inference_env.Agile_env_cfg_real import AgileDroneEnvCfgReal
from inference_env.utils import get_player_args

from neural_wbc.data import get_data_path

parser = get_player_args(description="Basic deployment player for running HOVER policy on real robots.")
args_cli = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    custom_config = None
    # if args_cli.env_config_overwrite is not None:
    #     with open(args_cli.env_config_overwrite) as fh:
    #         custom_config = yaml.safe_load(fh)
    #     print("[INFO]: Using custom configuration:")
    #     pprint.pprint(custom_config)

    env_cfg = AgileDroneEnvCfgReal()

    player = DeploymentPlayer(
        args_cli=args_cli,
        env_cfg=env_cfg,
        custom_config=custom_config,
    )

    inference_time = env_cfg.decimation * env_cfg.dt
    print("Deploying policy on real robot.")
    start_time = time.time()
    elapsed_time = 0.0
    for i in range(args_cli.max_iterations):
        print(
            "\rActual loop frequency: {:.2f} Hz | Update time: {:.2f}s".format(
                1 / (time.time() - start_time), elapsed_time
            ),
            end="",
            flush=True,
        )
        start_time = time.time()
        
        player.play_once()
        
        elapsed_time = time.time() - start_time
        remaining_time = inference_time - elapsed_time
        if remaining_time > 0.0:
            time.sleep(remaining_time)


if __name__ == "__main__":
    main()
