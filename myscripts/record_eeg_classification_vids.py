"""
Modified version of robosuite.demos.demo_video_recording

Record video of agent episodes with the imageio library.
This script uses offscreen rendering.

Example:
    $ python demo_video_recording.py --environment Lift --robots Panda
"""

import argparse
from operator import is_

import imageio
import numpy as np

import robosuite.utils.macros as macros
from robosuite import make
from robosuite.wrappers.visualization_wrapper import VisualizationWrapper
from robosuite.controllers import load_controller_config

import pandas as pd

import pdb
# Set the image convention to opencv so that the images are automatically rendered "right side up" when using imageio
# (which uses opencv convention)
macros.IMAGE_CONVENTION = "opencv"

np.random.seed(1)

def generate_action(target_pos, eef_pos, target_half_size, sample_good=True):
    """
    Generate action that are clearly good or bad

    Args:
        target_pos (array): target position. dimension should match eef_pos
        eef_pos (array): current eef position
        sample_good (bool): if true, returns "good" action. If false returns "bad" action

    Returns:
        action (array): "good" actions move eef straight towards the target
            "bad" actions clearly moves eef away from target 
    """
    u_good = (target_pos - eef_pos) / np.linalg.norm(target_pos - eef_pos) # unit vector in good direction (towards target)

    if sample_good:
        if is_in_target(target_pos, target_half_size, eef_pos):
            # if the eef is already in target region, good action is to not move
            action = np.zeros(2)
        else:
            action = 0.3 * (1 / np.max(np.abs(u_good))) * u_good # scale action to [-1,1] * 0.5
    
    else:
        theta = np.deg2rad(np.random.uniform(-45, 45, 1)[0]) # angle range
        rot = np.array([ # rotation matrix
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        u_bad = np.dot(rot, -u_good) # unit vector in bad direction (clearly away from target)
        action = 0.3 * (1 / np.max(np.abs(u_bad))) * u_bad
    
    return np.concatenate((action, np.array([0, -1])))

def is_good(p_good):
    if np.random.random() < p_good:
        return True
    return False

def is_in_target(target_pos, target_half_size, eef_pos):
    return (
        target_pos[0] - target_half_size[0] < eef_pos[0] < target_pos[0] + target_half_size[0]
        and target_pos[1] - target_half_size[1] < eef_pos[1] < target_pos[1] + target_half_size[1]
    )

if __name__ == "__main__":

    generate_action(np.array((0, 0)), np.array((1,1)), np.array((0.05, 0.05)), False)
    # pdb.set_trace()
    camera_names = "frontview"
    target_half_size = np.array((0.05, 0.05, 0.001))
    terminate_on_success = False

    # initialize an environment with offscreen renderer
    env = make(
        env_name="Reaching2D",
        controller_configs=load_controller_config(default_controller="OSC_POSITION"),
        robots="Panda",
        has_offscreen_renderer=True,
        render_camera="frontview",
        use_camera_obs=True,
        control_freq=20,
        target_half_size=target_half_size,
        random_init=True,
        random_target=True,
        has_renderer=False,
        # ignore_done=False,
        use_object_obs=False,
        camera_names=camera_names,
        camera_heights=512,
        camera_widths=512,
    )
    env = VisualizationWrapper(env, indicator_configs=None)

    timesteps = 500
    n_videos = 50
    p_good = 0.5 # probability of sampling good action
    switch_range = (30, 80) # sample new action every n steps n in (low, high)

    for vid in range(n_videos):

        good_hist = []
        eef_pos_hist = []  
        action_hist = []

        print(f"----------video {vid}----------")

        video_path = f"videos/video{vid}.mp4"

        # create a video writer with imageio
        writer = imageio.get_writer(video_path, fps=20)

        obs = env.reset()
        target_pos = env.target_pos

        # when to sample new action
        next_switch = np.random.randint(switch_range[0], switch_range[1])

        # sample first action
        good = is_good(p_good)
        action = generate_action(target_pos, obs["robot0_eef_pos"][:2], target_half_size[:2], good)
        print(f"First action is {action} (good = {good})")

        frames = []
        for i in range(timesteps):
            # print("step ", i)
            
            # if action is out of bounds, keep re-sampling action
            while not env._check_action_in_bounds(action):
                good = is_good(p_good)
                action = generate_action(target_pos, obs["robot0_eef_pos"][:2], target_half_size[:2], good)
                next_switch = i + np.random.randint(switch_range[0], switch_range[1])
                print(f"New action {action} (good = {good}) at step {i}")

            # if it is time to switch re-sample action
            if i >= next_switch or not env._check_action_in_bounds(action):

                good = is_good(p_good)
                action = generate_action(target_pos, obs["robot0_eef_pos"][:2], target_half_size[:2], good)
                next_switch = i + np.random.randint(switch_range[0], switch_range[1])
                print(f"New action {action} (good = {good}) at step {i}")

            # check if eef is in target
            if good == True and is_in_target(target_pos, 0.3*target_half_size[:2], obs["robot0_eef_pos"]):
                action = np.array([0, 0, 0, -1])

            obs, reward, done, info = env.step(action)

            frame = obs[camera_names + "_image"]
            writer.append_data(frame)

            # if i % 25 == 0:
            #     print("Saving frame #{}".format(i))

            ## TODO - log target position, time step, eef position, 
            good_hist.append(good)
            eef_pos_hist.append(obs["robot0_eef_pos"][:2])
            action_hist.append(action)      
            

            if terminate_on_success and done:
                break
        
        writer.close()
        
        # record target po, time, eef pos, actions
        df = pd.DataFrame(
            data={
                "step" : np.arange(len(good_hist)),
                "action" : pd.Series(action_hist),
                "eef_pos" : pd.Series(eef_pos_hist),
                "good" : good_hist,
                "target_pos" : pd.Series([target_pos for i in range(len(good_hist))]),
            }
        )
        df.to_csv(f"videos/video{vid}.csv", index=False)
        # pdb.set_trace()

