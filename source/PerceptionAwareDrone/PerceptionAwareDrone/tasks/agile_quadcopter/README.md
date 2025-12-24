1. Run teleop
```bash
    ros2 launch teleop_twist_joy teleop-launch.py
```

2. Run odor localization env
```bash
    python3 scripts/rl_games/velocity_control/odor_localization_env.py
```

3. Run Gaden simulation  
```bash     
    ros2 launch test_env main_simbot_2_sensor_launch.py
```