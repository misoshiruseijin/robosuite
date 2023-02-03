"""
Modified version of robosuite.demos.demo_video_recording

Record video of agent episodes with the imageio library.
This script uses offscreen rendering.

Example:
    $ python demo_video_recording.py --environment Lift --robots Panda
"""

import imageio
import numpy as np

import robosuite.macros as macros
from robosuite import make
from robosuite.wrappers.visualization_wrapper import VisualizationWrapper
from robosuite.controllers import load_controller_config

from robosuite.utils.primitive_skills import PrimitiveSkill

from robosuite.utils.input_utils import input2action
from robosuite.devices import SpaceMouse

import pandas as pd
import os
from pathlib import Path
from pynput import keyboard
import cv2
import shutil

import pdb
# Set the image convention to opencv so that the images are automatically rendered "right side up" when using imageio
# (which uses opencv convention)
macros.IMAGE_CONVENTION = "opencv"

np.random.seed(0)

def record_videos_2d_reaching(video_path="video.mp4", csv_path="data.csv", camera_names="frontview", good=True, timesteps=200):
    
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

    good_hist = []
    eef_pos_hist = []  
    action_hist = []

    # create a video writer with imageio
    writer = imageio.get_writer(video_path, fps=20)

    obs = env.reset()
    eef_pos = obs["robot0_eef_pos"]
    target_pos = env.target_pos
    bad_action_resample_cnt = 0

    # unit vector in direction of target
    u_good = (target_pos - eef_pos[:2]) / np.linalg.norm((target_pos - eef_pos[:2]))

    # if good, move straight to target. 
    if good:
        action = 0.3 * u_good

    # if bad, find bad action
    if not good:
        theta = np.deg2rad(np.random.uniform(-45, 45, 1)[0]) # angle range
        rot = np.array([ # rotation matrix
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        u_bad = np.dot(rot, -u_good) # unit vector in bad direction (clearly away from target)
        # action = 0.3 * u_bad
        action = 0.3 * -u_good

    action = np.concatenate([action, np.array([0, -1])])
    print("action ", action)

    for i in range(timesteps):

        # if good and target is reached, stop
        if good and env._check_success():
                action = np.array([0, 0, 0, -1])

        # if bad, go away from target. if action out of bounds, sample another bad action. if valid action is not found, stop
        if not good:
            while not env._check_action_in_bounds(action):
                theta = np.deg2rad(np.random.uniform(-80, 80, 1)[0]) # angle range
                rot = np.array([ # rotation matrix
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]
                ])
                u_bad = np.dot(rot, -u_good) # unit vector in bad direction (clearly away from target)
                action = 0.3 * u_bad
                action = np.concatenate([action, np.array([0, -1])])
                bad_action_resample_cnt += 1
                if bad_action_resample_cnt >= 50:
                    # if unable to find valid action, zero action
                    action = np.array([0, 0, 0, -1])
                    print("max resample")
                    break
        
        obs, reward, done, info = env.step(action)

        frame = obs[camera_names + "_image"]
        writer.append_data(frame)

        # log target position, action, eef position, 
        # print("action ", action)
        good_hist.append(good)
        eef_pos_hist.append(obs["robot0_eef_pos"][:2])
        action_hist.append(action)   
        # print("reward ", reward)
        # pdb.set_trace()   
    
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
    df.to_csv(csv_path, index=False)



################## Franka 2D Reaching ###########################
def is_good(p_good):
    if np.random.random() < p_good:
        return True
    return False

def is_in_target(target_pos, target_half_size, eef_pos):
    
    return (
        target_pos[0] - target_half_size[0] < eef_pos[0] < target_pos[0] + target_half_size[0]
        and target_pos[1] - target_half_size[1] < eef_pos[1] < target_pos[1] + target_half_size[1]
    )

def record_videos():
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
                action = 0.3 * (1 / np.max(np.abs(u_good))) * u_good # scale action
        
        else:
            theta = np.deg2rad(np.random.uniform(-45, 45, 1)[0]) # angle range
            rot = np.array([ # rotation matrix
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            u_bad = np.dot(rot, -u_good) # unit vector in bad direction (clearly away from target)
            action = 0.3 * (1 / np.max(np.abs(u_bad))) * u_bad
        
        return np.concatenate((action, np.array([0, -1])))
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

def trim_video(vid_path, csv_path, save_folder, use_prefix=False, len_thresh=30):
    import skvideo.io

    # read video and csv
    vid = skvideo.io.vread(vid_path)
    df = pd.read_csv(csv_path)
    goods = df["good"]
    start_frame = 0
    clip = 0

    csv_save_path = os.path.join(save_folder, "csvs")
    vid_save_path = os.path.join(save_folder, "mp4s")
    os.makedirs(csv_save_path, exist_ok=True)
    os.makedirs(vid_save_path, exist_ok=True)

    for i in range(1, len(goods)):
        if not (goods[i] == goods[i-1]): # if behavior changed
            clip_len = i - start_frame
            if clip_len >= len_thresh: # skip if clip is too short

                # cut csv and video, save them
                vid_name = f"clip{clip}_len{clip_len}.mp4"
                csv_name = f"clip{clip}_len{clip_len}.csv"
                
                if use_prefix:
                    vid_name = Path(vid_path).stem + "_" + vid_name
                    csv_name = Path(vid_path).stem + "_" + csv_name

                skvideo.io.vwrite(os.path.join(vid_save_path, vid_name), vid[start_frame:i])
                df_clip = df[start_frame:i]
                df_clip["step"] = np.arange(len(df_clip["step"]))
                df_clip.to_csv(os.path.join(csv_save_path, csv_name), index=False)
                clip += 1

            start_frame = i

    # write last clip
    clip_len = len(goods) - start_frame
    if clip_len >= len_thresh:
        vid_name = f"clip{clip}_len{clip_len}.mp4"
        csv_name = f"clip{clip}_len{clip_len}.csv"

        if use_prefix:
            vid_name = Path(vid_path).stem + "_" + vid_name
            csv_name = Path(vid_path).stem + "_" + csv_name
        
        skvideo.io.vwrite(os.path.join(vid_save_path, vid_name), vid[start_frame:])
        df_clip = df[start_frame:]
        df_clip["step"] = np.arange(len(df_clip["step"]))
        df_clip.to_csv(os.path.join(csv_save_path, csv_name), index=False)
        df[start_frame:].to_csv(os.path.join(csv_save_path, csv_name), index=False)

def liststr_to_list(liststr, delimiter):
    from ast import literal_eval
    liststr = liststr[:1] + liststr[2:]
    delimiter_idx = liststr.find(delimiter)
    return literal_eval(liststr[:delimiter_idx] + "," + liststr[delimiter_idx:])

def add_eef_in_target_to_csv(csv_path, target_half_size=(0.05, 0.05)):
    
    df = pd.read_csv(csv_path)
    target_pos = liststr_to_list(df["target_pos"][0], delimiter=" ")
    in_target = []

    for elem in df["eef_pos"]:
        eef_pos = liststr_to_list(elem, delimiter=" ")
        if is_in_target(target_pos, target_half_size, eef_pos):
            in_target.append(True)
        else:
            in_target.append(False)

    df["in_target"] = in_target
    df.to_csv(csv_path, index=False)

################## Franka 2D Reaching ###########################

################## Franka Drop ##################################
def generate_action_drop(drop_pos, stage_half_size, eef_pos, sample_good):
    """
    drop_pos: good position to drop object (somewhere above stage)
    sample_good: whether generated action should be good
    """
    speed = 0.3
    dropped = False # whether this action drops the object

    # if eef height is not enough, move straight up
    if eef_pos[2] < drop_pos[2]: 
        action = speed * np.array([0, 0, 1, 1])
        action[-1] = 1 # keep gripper closed

    else:
        eps = 0.02
        should_drop = (
            (drop_pos[0] - stage_half_size[0] + eps < eef_pos[0] < drop_pos[0] + stage_half_size[0] - eps)
                and (drop_pos[1] - stage_half_size[1] + eps < eef_pos[1] < drop_pos[1] + stage_half_size[1] - eps)
        )

        good_u = (drop_pos - eef_pos) / np.linalg.norm(drop_pos - eef_pos) # good direction to move in

        if sample_good:
            # if good action is requested
            if should_drop:
                # drop
                action = np.array([0, 0, 0, -1])
                dropped = True
            else:
                # if not in good position to drop
                action = speed * good_u
                action = np.append(action, 1) # keep gripper closed

        else:
            # bad action is requested (should_drop should never be true)
            theta = np.deg2rad(np.random.uniform(-45, 45, 1)[0]) # angle range
            rot = np.array([ # rotation matrix
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            u_bad = np.dot(rot, -good_u) # unit vector in bad direction (clearly away from target)
            action = 0.3 * (1 / np.max(np.abs(u_bad))) * u_bad
            action = np.append(action, 1)
    
    return action, dropped

def record_videos_drop():
    camera_names = "frontview"
    terminate_on_success = False
    # initialize an environment with offscreen renderer
    env = make(
        env_name="Drop",
        controller_configs=load_controller_config(default_controller="OSC_POSITION"),
        robots="Panda",
        has_offscreen_renderer=True,
        render_camera="frontview",
        use_camera_obs=True,
        control_freq=20,
        random_init=True,
        random_stage=True,
        stage_type="basket",
        has_renderer=False,
        use_object_obs=False,
        camera_names=camera_names,
        camera_heights=512,
        camera_widths=512,
    )
    env = VisualizationWrapper(env, indicator_configs=None)

    # timesteps = 500
    n_videos = 50
    p_good = 0.5 # probability of sampling good action
    # stage_half_size = env.stage_half_size

    for vid in range(n_videos):
        """
        - choose good or bad
        - get eef and stage surface position
        - if start eef position is below the stage, move straight up 

        Good Case:
        - move to above stage, then release

        Bad Case:
        - move to somewhere other than above stage, then release
        """

        # good_hist = []
        eef_state_hist = [] # x, y, z, gripper 
        action_hist = []
        success = []
        failure = []

        print(f"----------video {vid}----------")

        video_path = f"videos/video{vid}.mp4"

        # create a video writer with imageio
        writer = imageio.get_writer(video_path, fps=20)

        obs = env.reset()
        stage_pos = env.stage_top
        stage_half_size = env.stage_half_size
        drop_pos = stage_pos + np.array([0, 0, 0.08])
        dropped = False

        # choose if this clip is good
        good = is_good(p_good)
        print("Good: ", good)
        done = False
        steps_after_drop = 30 # how many steps to record after drop action is taken

        # pdb.set_trace()
        while steps_after_drop > 0:
            
            if not dropped:
                action, dropped = generate_action_drop(drop_pos, stage_half_size, obs["robot0_eef_pos"], good)
            
            # if action is out of bounds, drop
            if not env._check_action_in_bounds(action):
                action = np.array([0, 0, 0, -1])
                dropped = True

            obs, reward, done, info = env.step(action)

            frame = obs[camera_names + "_image"]
            writer.append_data(frame)

            eef_state_hist.append(obs["robot0_eef_pos"])
            action_hist.append(action)

            if dropped and good:
                success.append(True)
            else:
                success.append(False)
            if dropped and not good:
                failure.append(True)
            else:
                failure.append(False)

            if done:
                steps_after_drop -= 1
        
        writer.close()
        
        # record target po, time, eef pos, actions
        df = pd.DataFrame(
            data={
                "step" : np.arange(len(action_hist)),
                "action" : pd.Series(action_hist),
                "eef_pos" : pd.Series(eef_state_hist),
                "good" : [good] * len(action_hist),
                "drop_pos" : pd.Series([drop_pos] * len(action_hist)),
                "success": success,
                "failure": failure,
            }
        )
        df.to_csv(f"videos/video{vid}.csv", index=False)

################# Franka Cube Slide #####################
def record_videos_move_obj(initial_eef_pos, final_eef_pos, speed, video_path="video.mp4", csv_path="data.csv", camera_names="frontview", obj_rgba=(1,0,0,1)):

    # initialize an environment with offscreen renderer
    env = make(
        env_name="Object_and_Table",
        controller_configs=load_controller_config(default_controller="OSC_POSITION"),
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        use_camera_obs=True,
        control_freq=20,
        ignore_done=True,
        initial_eef_pos=initial_eef_pos, # where on the table eef should start from
        obj_half_size=(0.025, 0.025, 0.025), # half size of object (cube)
        obj_rgba=obj_rgba, # object rgba 
        camera_names=camera_names,
        camera_heights=512,
        camera_widths=512,
    )

    obs = env.reset()
    eef_pos = obs["robot0_eef_pos"]

    # create a video writer with imageio
    writer = imageio.get_writer(video_path, fps=20)

    thresh = 0.03 # how close eef must be before recording stops
    if speed > 0.9:
        thresh = 0.1

    eef_pos_hist = []
    action_hist = []

    final_eef_pos = final_eef_pos + np.array([0, 0, env.table_offset[2] + 0.04])
    initial_eef_pos = initial_eef_pos + np.array([0, 0, env.table_offset[2] + 0.04])

    n_frames = 0

    while np.any(np.abs(eef_pos - final_eef_pos) > thresh):
        if n_frames % 100 == 0:
            print("frame ", n_frames)
        u = (final_eef_pos - eef_pos) / np.linalg.norm(final_eef_pos - eef_pos)
        action = np.append(speed * u, 1)
        obs, reward, done, info = env.step(action)
        frame = obs[camera_names + "_image"]
        writer.append_data(frame)

        eef_pos = obs["robot0_eef_pos"]
        eef_pos_hist.append(eef_pos)
        action_hist.append(action)
        # print("action ", action)
        # print("eef pos ", eef_pos)
        # print("error ", eef_pos - final_eef_pos)
        n_frames += 1

    writer.close()

    df = pd.DataFrame(
        data={
            "step" : np.arange(len(action_hist)),
            "action" : pd.Series(action_hist),
            "eef_pos" : pd.Series(eef_pos_hist),
        }
    )
    df.to_csv(csv_path, index=False)

################### Franka Lift #########################
def record_videos_lift(video_path="video.mp4", csv_path="data.csv", camera_names="frontview", obj_rgba=(1,0,0,1), good=True):

    def generate_action_lift(target_pos, closed_gripper):
        """
        Args:
            target_pos: good position to pick object up
            eef_pos: current eef position

        Returns:
            action
        """
        thresh = 0.01
        action_scale = 0.5
        # goal_pos = target_pos + np.array([0, 0, -0.01])

        # if not good:
        #     # if action should be "bad", add random offset to goal position
        #     rand_direction = np.concatenate((np.random.uniform(-1, 1, 2), np.random.uniform(0, 1, 1)))
        #     rand_direction = rand_direction / np.linalg.norm(rand_direction)
        #     goal_pos = target_pos + 0.03 * rand_direction
        
        closed = closed_gripper

        u = (goal_pos - eef_pos) / np.linalg.norm(goal_pos - eef_pos)

        if closed_gripper:
            # lift
            action = np.array([0, 0, 0.25, 0, 0, 0, 1])

        elif np.all(np.abs(goal_pos - eef_pos) < thresh):
            # grasp
            action = np.array([0, 0, 0, 0, 0, 0, 1])
            closed = True

        else:
            # move towards goal
            action = np.concatenate((action_scale*u, np.array([0, 0, 0, -1])))
            
        return action, closed

    # initialize an environment with offscreen renderer
    env = make(
        env_name="Lift2",
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        use_camera_obs=True,
        control_freq=20,
        ignore_done=False,
        camera_names=camera_names,
        camera_heights=512,
        camera_widths=512,
        obj_rgba=obj_rgba,
    )

    obs = env.reset()
    eef_pos = obs["robot0_eef_pos"]

    # create a video writer with imageio
    writer = imageio.get_writer(video_path, fps=20)

    eef_pos_hist = []
    action_hist = []
    obj_pos_hist = []
    success_hist = []

    # get object initial position
    obj_pos = obs["cube_pos"]
    goal_pos = obj_pos
    if good:
        goal_pos = goal_pos + np.array([0, 0, -0.005])
    else:
        rand_direction = np.concatenate((np.random.uniform(-1, 1, 2), np.random.uniform(0.5, 1, 1)))
        rand_direction = rand_direction / np.linalg.norm(rand_direction)
        goal_pos = goal_pos + 0.03 * rand_direction

    success = False
    closed = False

    n_frames = 0
    lift_steps = 0
    stop_steps = 0
    lift_done = lift_steps >= 30

    while stop_steps < 30: # done checks for success, lift_steps make sure recording stops for failure case
        # print("Stop steps ", stop_steps)
        # print("lift_steps ", lift_steps)
        # print("done ", done)
        if n_frames % 100 == 0:
            print("frame ", n_frames)
        
        if success or lift_done:
            # record a few extra frames after lift is completed
            action = np.array([0, 0, 0, 0, 0, 0, 1])
            closed = True
            stop_steps += 1

        else:
            action, closed = generate_action_lift(goal_pos, closed)

        obs, reward, done, info = env.step(action)
        frame = obs[camera_names + "_image"]
        writer.append_data(frame)

        eef_pos = obs["robot0_eef_pos"]
        eef_pos_hist.append(eef_pos)
        action_hist.append(action)
        obj_pos_hist.append(obs["cube_pos"])
        success = reward > 0
        success_hist.append(success)

        # print("action ", action)
        # print("eef pos ", eef_pos)
        # print("error ", eef_pos - final_eef_pos)
        n_frames += 1
        lift_steps += closed
        lift_done = lift_steps >= 30
        if n_frames > 400:
            # something is wrong. stop
            break

    writer.close()

    df = pd.DataFrame(
        data={
            "step" : np.arange(len(action_hist)),
            "action" : pd.Series(action_hist),
            "eef_pos" : pd.Series(eef_pos_hist),
            "obj_pos" : pd.Series(obj_pos_hist),
            "success" : pd.Series(success_hist),
        }
    )
    df.to_csv(csv_path, index=False)

################## Abstract States Stimulus ##############
def record_video_abstract_states(start_state, end_state, video_path="video.mp4", csv_path="data.csv", camera_names="frontview", obj_rgba=(1,0,0,1), speed=0.25):

    assert (1 <= start_state <= 9 and 1 <= end_state <= 9)

    env = make(
        env_name="GridWall",
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        use_camera_obs=True,
        control_freq=20,
        ignore_done=False,
        camera_names=camera_names,
        camera_heights=512,
        camera_widths=512,
        obj_rgba=obj_rgba,
        obj_intial_abs_state=start_state,
    )

    obs = env.reset()
    eef_pos = obs["robot0_eef_pos"]
    obj_pos = obs["obj_pos"]
    lift_height = eef_pos[2]
    pick_place_height = env.table_offset[2] + env.obj_half_size[2]
    goal_pos = env.abstract_states[end_state]
    thresh = 0.005

    # create a video writer with imageio
    writer = imageio.get_writer(video_path, fps=20)

    eef_pos_hist = []
    action_hist = []
    obj_pos_hist = []
    eef_state_hist = []
    obj_state_hist = []

    n_frames = 0
    lift_steps = 0
    stop_steps = 0
    gripper_cnt = 0 # buffer to make sure object is gripped

    phase = 1

    while stop_steps < 30: # done checks for success, lift_steps make sure recording stops for failure case
        """
        phases
        1. move to cube initial position
        2. move down
        3. grip
        4. move up
        5. move to final position
        6. move down, release
        7. move up
        8. stop
        """

        if n_frames % 100 == 0:
            print("frame ", n_frames)

        if phase == 1: # reach to pick pos
            u = (obj_pos[:2] - eef_pos[:2]) / np.linalg.norm(obj_pos[:2] - eef_pos[:2])
            action = 1.5 * speed * u
            action = np.array([action[0], action[1], 0, 0, 0, 0, -1])
            if np.all(np.abs(eef_pos[:2] - obj_pos[:2]) < thresh):
                phase = 2
        if phase == 2: # move down (open)
            action = np.array([0, 0, -speed, 0, 0, 0, -1])
            if np.abs(eef_pos[2] - pick_place_height) < thresh:
                phase = 3
        if phase == 3: # grip
            action = np.array([0, 0, 0, 0, 0, 0, 1])
            gripper_cnt += 1
            if gripper_cnt > 15:
                gripper_cnt = 0
                phase = 4
        if phase == 4: # move up (close)
            action = np.array([0, 0, speed, 0, 0, 0, 1])
            if np.abs(eef_pos[2] - lift_height) < thresh:
                phase = 5
        if phase == 5: # reach to drop pos
            u = (goal_pos[:2] - eef_pos[:2]) / np.linalg.norm(goal_pos[:2] - eef_pos[:2])
            action = 1.5 * speed * u
            action = np.array([action[0], action[1], 0, 0, 0, 0, 1])
            if np.all(np.abs(eef_pos[:2] - goal_pos[:2]) < thresh):
                phase = 6
        if phase == 6: # move down (close)
            action = np.array([0, 0, -speed, 0, 0, 0, 1])
            if abs(eef_pos[2] - pick_place_height) < thresh:
                phase = 7
        if phase == 7: # drop
            action = np.array([0, 0, 0, 0, 0, 0, -1])
            gripper_cnt += 1
            if gripper_cnt >= 15:
                gripper_cnt = 0
                phase = 8
        if phase == 8: # move up (open)
            action = np.array([0, 0, speed, 0, 0, 0, -1])
            if np.abs(eef_pos[2] - lift_height) < thresh:
                # stop_steps += 1
                break

        obs, reward, done, info = env.step(action)
        frame = obs[camera_names + "_image"]
        writer.append_data(frame)

        eef_pos = obs["robot0_eef_pos"]
        eef_pos_hist.append(obs["eef_xyz_gripper"])
        action_hist.append(action)
        obj_pos_hist.append(obs["obj_pos"])
        eef_state_hist.append(np.where(obs["eef_abstract_state"] == True)[0][0] + 1)
        obj_state_hist.append(np.where(obs["obj_abstract_state"] == True)[0][0] + 1)

        # print("action ", action)
        # print("phase ", phase)
        # print("eef pos ", eef_pos)
        # print("goal ", goal_pos)
        # print("obj pos ", obj_pos)

        n_frames += 1
        if n_frames > 500:
            # something is wrong. stop
            break

    writer.close()

    df = pd.DataFrame(
        data={
            "step" : np.arange(len(action_hist)),
            "action" : pd.Series(action_hist),
            "eef_pos" : pd.Series(eef_pos_hist),
            "obj_pos" : pd.Series(obj_pos_hist),
            "eef_abstract_state" : pd.Series(eef_state_hist),
            "obj_abstract_state" : pd.Series(obj_state_hist),
            "grid_half_size" : [env.grid_half_size] * len(action_hist)
        }
    )
    df.to_csv(csv_path, index=False)

################## Drawer Open/Close Stimulus ###############
def record_video_drawer(video_path="video.mp4", csv_path="data.csv", camera_names="frontview", thresh=0.001, speed=0.1):

    env = make(
        env_name="DrawerEnv",
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        use_camera_obs=True,
        control_freq=20,
        ignore_done=False,
        camera_names=camera_names,
        camera_heights=512,
        camera_widths=512,
    )

    obs = env.reset()
    eef_pos = obs["robot0_eef_pos"]

    grip_height = 1.022
    zero_pull_y = 0.017
    full_pull_y = -0.304

    phase = 1

    # create a video writer with imageio
    writer = imageio.get_writer(video_path, fps=20)

    eef_pos_hist = []
    action_hist = []
    drawer_pos_hist = []

    n_frames = 0
    pre_record_steps = 0
    post_record_steps = 0
    transition_steps = 0
    steps_to_idle = 50
    idle_phase = 1
    recording = False

    phase = 1
    # 1 (down) -> 2 (grip) -> 3 (close) -start recording-> 5 (idle) -> 4 (open) -> 5 idle -> 3 close -> 5 idle -> stop recording  
    while post_record_steps < steps_to_idle: # done checks for success, lift_steps make sure recording stops for failure case

        if not recording: 
            print("initializing...")
            print("phase", phase)
            print("eef pos ", eef_pos)
        elif n_frames % 100 == 0:
            print("frame ", n_frames)

        # get action
        if phase == 1: # down
            action = np.array([0, 0, -0.1, 0, 0, 0, -1])
            if np.abs(obs["eef_xyz_gripper"][2] - grip_height) < thresh:
                phase = 2
        if phase == 2: # grip
            action = np.array([0, 0, 0, 0, 0, 0, 1])
            if obs["eef_xyz_gripper"][-1] == 1:
                phase = 3
        if phase == 3: # close
            action = np.array([0, speed, 0, 0, 0, 0, 1])
            if np.abs(obs["eef_xyz_gripper"][1] - zero_pull_y) < thresh:
                phase = 5
                recording = True
        if phase == 4: # open
            action = np.array([0, -speed, 0, 0, 0, 0, 1])
            if np.abs(obs["eef_xyz_gripper"][1] - full_pull_y) < thresh:
                phase = 5
        if phase == 5: # idle
            action = np.array([0, 0, 0, 0, 0, 0, 1])
            if idle_phase == 1:
                pre_record_steps += 1
                if pre_record_steps >= steps_to_idle: # pre-recording
                    phase = 4
                    idle_phase += 1
            elif idle_phase == 2: # between open and close
                transition_steps += 1
                if transition_steps >= steps_to_idle:
                    phase = 3
                    idle_phase += 1
            elif idle_phase == 3:
                post_record_steps += 1

        obs, reward, done, info = env.step(action)
        eef_pos = obs["robot0_eef_pos"]
        drawer_pos = eef_pos[1] - zero_pull_y

        if recording:
            frame = obs[camera_names + "_image"]
            writer.append_data(frame)

            eef_pos_hist.append(obs["eef_xyz_gripper"])
            action_hist.append(action)
            drawer_pos_hist.append(drawer_pos)

            # print("action ", action)
            # print("phase ", phase)
            # print("eef pos ", eef_pos)
            print("drawer pos ", eef_pos[1] - zero_pull_y)

            n_frames += 1
            if n_frames > 1500:
                # something is wrong. stop
                break

    writer.close()

    df = pd.DataFrame(
        data={
            "step" : np.arange(len(action_hist)),
            "action" : pd.Series(action_hist),
            "eef_pos" : pd.Series(eef_pos_hist),
            "drawer_pos " : pd.Series(drawer_pos_hist),
        }
    )
    df.to_csv(csv_path, index=False)

################## Roll Ball Left Right Stimulus ############
def record_left_right(video_path="video.mp4", camera_names="frontview", ball_rgba=(1,0,0,1)):
    # initialize environment with offscreen renderer
    env = make(
        env_name="LeftRight",
        robots="Panda",
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        initialization_noise=None,
        table_full_size=(0.8, 2.0, 0.01),
        use_camera_obs=True,
        use_object_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        ignore_done=True,
        camera_names=camera_names,
        camera_heights=512,
        camera_widths=512,
        line_thickness=0.02,
        line_rgba=(0,0,0,1),
        ball_radius=0.04,
        ball_rgba=ball_rgba
    )

    obs = env.reset()

    action_hist = []
    eef_pos_hist = []
    ball_pos_hist = []

    # create a video writer with imageio
    writer = imageio.get_writer(video_path, fps=20)

    # small random movements to randomize ball roll direction for first few steps, then release
    random_steps = 25
    total_steps = 100

    for i in range(total_steps):
        if i < random_steps:
            action = np.random.uniform(-1, 1, 6)
            action[:3] *= 0.05
            action[3:] *= 0.1
            action = np.concatenate([action, np.array([1])])
        else:
            action = np.array([0, 0, 0, 0, 0, 0, -1])

        obs, _, _, _ = env.step(action)
        frame = obs[camera_names + "_image"]
        writer.append_data(frame)

        action_hist.append(action)
        eef_pos_hist.append(obs["robot0_eef_pos"])
        ball_pos_hist.append(obs["ball_pos"])

    writer.close()

    # check which side ball ended up in
    if ball_pos_hist[-1][1] < 0:
        ball_side = "left"
    else:
        ball_side = "right"

    # save datafile
    # n = len(action_hist)
    # df = pd.DataFrame(
    #     data={
    #         "step" : np.arange(n),
    #         "action" : pd.Series(action_hist),
    #         "eef_pos" : pd.Series(eef_pos_hist),
    #         "ball_pos" : pd.Series(ball_pos_hist),
    #         "ball_ended_up_on" : pd.Series([ball_side] * n),
    #         "ball_goal_side" : pd.Series([goal] * n)
    #     }
    # )
    # df.to_csv(csv_path, index=False)

    return action_hist, eef_pos_hist, ball_pos_hist, ball_side

####### TEST PRIMITIVE #########
def test_primitive():
    # initialize an environment with offscreen renderer
    camera_names = "frontview"
    target_half_size = np.array((0.05, 0.05, 0.001))
    video_path = "video.mp4"
    csv_path = "data.csv"

    obs_hist = []
    reward_hist = []
    done_hist = []
    info_hist = []

    # initialize an environment with offscreen renderer
    env = make(
        env_name="Lift2",
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        use_camera_obs=True,
        control_freq=20,
        ignore_done=False,
        camera_names=camera_names,
        camera_heights=512,
        camera_widths=512,
        obj_rgba=(1,0,0,1),
    )

    env = VisualizationWrapper(env, indicator_configs=None)
    obs = env.reset()
    # create a video writer with imageio
    writer = imageio.get_writer(video_path, fps=20)

    primitive = PrimitiveSkill(env, return_all_states=True)
    observations, rewards, dones, infos = primitive.move_to_pos(
        obs=obs,
        goal_pos=(0.16,0.16,0.988),
        gripper_closed=False,
        robot_id=0,
        speed=0.25
    )

    for ob in observations:
        frame = ob[camera_names + "_image"]
        writer.append_data(frame)

    writer.close()

########## Spacemouse Control ############
def spacemouse_control(env, obs_to_print=["robot0_eef_pos"], indicator_on=True):
    
    if indicator_on:
        env = VisualizationWrapper(env, indicator_configs=None)

    controller_type = env.robot_configs[0]["controller_config"]["type"]
    if controller_type == "OSC_POSE":
        action_dim = 6
    elif controller_type == "OSC_POSITION":
        action_dim = 3
    
    device = SpaceMouse(
            vendor_id=9583,
            product_id=50734,
            pos_sensitivity=1.0,
            rot_sensitivity=1.0,
        )

    device.start_control()
    while True:
        # Reset environment
        obs = env.reset()
        env.modify_observable(observable_name="robot0_joint_pos", attribute="active", modifier=True)

        # rendering setup
        env.render()

        # Initialize device control
        device.start_control()

        while True:
            # set active robot
            active_robot = env.robots[0]
            # get action
            action, grip = input2action(
                device=device,
                robot=active_robot,
            )
            if action_dim < 6:
                action = np.append(action[:action_dim], action[-1])

            if action is None:
                break

            # take step in simulation
            obs, reward, done, info = env.step(action)

            for ob in obs_to_print:
                print(f"{ob}: {obs[ob]}")

            env.render()

########## Manual Record Start/Stop ###########
# global flag to be accessed by on_press callback - TODO: is there way to avoid global variables?
recording = False
def manual_record(env, video_path="video.mp4", csv_path="data.csv", camera_names="frontview"):

    waypoints=[(0.16, 0.16, 0.988), (0.16, -0.16, 0.988)]
    goal_idx = 0
    goal_pos = waypoints[goal_idx]
    thresh = 0.002
    speed = 0.15
    obs = env.reset()
    eef_pos = obs["robot0_eef_pos"]
    error = goal_pos - eef_pos
    started_recording = False
    finished_recording = False
    writer = imageio.get_writer(video_path, fps=20)
    global recording

    def on_press(key):
        if key.char == "r":
            global recording
            recording = not recording
            if recording:
                print("Recording....")
            else: 
                print("Stopped recording")

    def on_release(key):
        return True

    # start non-blocking listener:
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()

    while np.any(np.abs(error)) > thresh:

        if not started_recording and recording:
            started_recording = True
            print("started ", started_recording)
        if started_recording and not recording:
            finished_recording = True
            print("finished ", finished_recording)
            break

        action = speed * error / np.linalg.norm(error)
        action = np.concatenate([action, np.array([0, 0, 0, -1])])
        obs, reward, done, info = env.step(action)
        frame = obs[camera_names + "_image"]
        # pdb.set_trace()
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
        
        # pdb.set_trace()
        if started_recording:
            writer.append_data(frame)
        
        if np.all(np.abs(error)) <= thresh:
            goal_idx = not goal_idx
            goal_pos = waypoints[goal_idx]

    writer.close()

def manual_record_spacemouse(env, device, video_path="video.mp4", camera_names=["agentview"]):
    
    """
    Record video manually using spacemouse. Start recording by pressing "r", stop recording by pressing "r" again.

    Args:
        env: environment
        video_path: location to save recorded video
        camera_names: simulation camera view

    Returns:
        action_hist (list): list of actions taken
        eef_pos_hist (list): list of end effector positions encountered 
        reward_hist (list): list of rewards earned 
    """

    global recording
    recording = False    
    # record start/stop keypress callback
    def on_press(key):
        if key.char == "r":
            global recording
            recording = not recording
            if recording:
                print("Recording....")
            else: 
                print("Stopped recording")

    def on_release(key):
        return True

    # start non-blocking listener:
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()

    # get action dimension
    controller_type = env.robot_configs[0]["controller_config"]["type"]
    if controller_type == "OSC_POSE":
        action_dim = 6
    elif controller_type == "OSC_POSITION":
        action_dim = 3

    active_robot = env.robots[0]
    obs = env.reset()
    started_recording = False
    finished_recording = False

    # initialize list of writers for each camera view
    writers = []
    for view in camera_names:
        writers.append(
            imageio.get_writer(video_path[:video_path.find(".")] + f"_{view}" + video_path[video_path.find("."):], fps=20)
        )
    
    eef_pos_hist = []
    action_hist = []
    reward_hist = []

    while True:
        print(recording)
        # update recording state
        if not started_recording and recording:
            started_recording = True
            print("started ", started_recording)
        if started_recording and not recording:
            finished_recording = True
            print("finished ", finished_recording)
            recording = False
            break

        # get action from
        action, grip = input2action(device=device, robot=active_robot)
        if action_dim < 6:
            action = np.append(action[:action_dim], action[-1])
        if action is None:
            break
        obs, reward, done, info = env.step(action)
        render_frame = obs[camera_names[0] + "_image"]
        # pdb.set_trace()
        cv2.imshow("frame", cv2.cvtColor(render_frame, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)
        
        # pdb.set_trace()
        if started_recording:
            for writer, view in zip(writers, camera_names):
                writer.append_data(obs[view + "_image"])
            action_hist.append(action)
            eef_pos_hist.append(obs["robot0_eef_pos"])
            reward_hist.append(reward)

    # close video writers and keyboard listener
    listener.stop()
    for writer in writers:
        writer.close()

    return action_hist, eef_pos_hist, reward_hist

################### Flashing Cube Pick and Place #########################
###OLD###
def record_cube_flash(video_path="video.mp4", camera_names="frontview2", cube_rgba=(0,0,0,1), led_color="white", change_color_every_steps=1, fps=20):
    # initialize environment with offscreen renderer
    env = make(
        env_name="LiftFlash",
        robots="Panda",
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        initialization_noise=None,
        table_full_size=(0.8, 2.0, 0.01),
        use_camera_obs=True,
        use_object_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview2",
        ignore_done=True,
        camera_names=camera_names,
        camera_heights=512,
        camera_widths=512,
        cube_rgba=cube_rgba,
        led_color=led_color,
        change_color_every_steps=change_color_every_steps,
    )

    p = PrimitiveSkill(env, return_all_states=True)
    # p.move_to_external_call = True
    
    obs = env.reset()

    # create a video writer with imageio
    writer = imageio.get_writer(video_path, fps=fps)

    # move to above cube
    print("[1] move to above pos")
    obs, reward, done, info = p.move_to_pos(
        obs=obs,
        goal_pos=(0, 0, p.home_pos[2]),
        gripper_closed=False
    )
    for ob in obs:
        frame = ob[camera_names + "_image"]
        writer.append_data(frame)

    print("[2] move to pick pos")
    # move down
    obs, reward, done, info = p.move_to_pos(
        obs=obs[-1],
        goal_pos=(0, 0, env.table_offset[2]+env.cube_half_size[2]),
        gripper_closed=False
    )
    for ob in obs:
        frame = ob[camera_names + "_image"]
        writer.append_data(frame)

    print("[3] grip")
    # grip
    for _ in range(15):
        action = np.array([0, 0, 0, 0, 0, 0, 1])
        obs, reward, done, info = env.step(action)
        frame = obs[camera_names + "_image"]
        writer.append_data(frame)

    print("[4] move to above pos")
    # pdb.set_trace()
    # move up
    obs, reward, done, info = p.move_to_pos(
        obs=obs,
        goal_pos=(0, 0, p.home_pos[2]),
        gripper_closed=True
    )
    for ob in obs:
        frame = ob[camera_names + "_image"]
        writer.append_data(frame)

    # pdb.set_trace()
    print("[5] move down")
    # move down
    obs, reward, done, info = p.move_to_pos(
        obs=obs[-1],
        goal_pos=(0, 0, env.table_offset[2]+env.cube_half_size[2]),
        gripper_closed=True
    )
    for ob in obs:
        frame = ob[camera_names + "_image"]
        writer.append_data(frame)

    print("[6] release")
    # release
    for _ in range(15):
        action = np.array([0, 0, 0, 0, 0, 0, -1])
        obs, reward, done, info = env.step(action)
        frame = obs[camera_names + "_image"]
        writer.append_data(frame)

    print("[7] move to above pos")
    # move up
    obs, reward, done, info = p.move_to_pos(
        obs=obs,
        goal_pos=(0, 0, p.home_pos[2]),
        gripper_closed=False
    )
    for ob in obs:
        frame = ob[camera_names + "_image"]
        writer.append_data(frame)
        
    writer.close()

    # return action_hist, eef_pos_hist, ball_pos_hist, ball_side

import threading
import time
def flash_cube(env, freq=2):
    interval = 0.5 / freq
    prev_time = time.time()
    while True:
        cur_time = time.time()
        if cur_time - prev_time >= interval:
            env._switch_led_on_off()
            prev_time = cur_time

def record_cube_flash(cube_rgba=(0,0,0,1), led_color="white", flash_freq=6):
    
    env = make(
        env_name="LiftFlash",
        robots="Panda",
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        use_camera_obs=False,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        render_camera="frontview2",
        camera_names="frontview2",
        cube_rgba=cube_rgba,
        led_color=led_color,
    )

    obs = env.reset()
    env.render()
    print("------Manually start recording the render window-------")
    pdb.set_trace()

    prev_time = time.time()
    interval = 0.5 / flash_freq # 2 Hz
    p = PrimitiveSkill()

    # do pick and place
    skill_done = False
    initial_cube_pos = obs["cube_pos"]
    goal = np.append(initial_cube_pos, 0)
    print("pick")
    while not skill_done:
        cur_time = time.time()
        if cur_time - prev_time >= interval:
            env._switch_led_on_off()
            prev_time = cur_time
        action, skill_done, failed = p._pick(obs=obs, params=goal)
        obs, reward, done, info = env.step(action)
        env.render()
    
    print("place")
    skill_done = False
    while not skill_done:
        cur_time = time.time()
        if cur_time - prev_time >= interval:
            env._switch_led_on_off()
            prev_time = cur_time
        action, skill_done, failed = p._place(obs=obs, params=goal)
        obs, reward, done, info = env.step(action)
        env.render()

if __name__ == "__main__":

    colors = {
        "red" : (1,0,0,1),
        "green" : (0,1,0,1),
        "blue" : (0,0,1,1),
        "yellow" : (1,1,0,1),
        "gray" : (0.75,0.75,0.75,1),
    }


    # ############## Flashing Cube ###############
    # record_cube_flash()


    ############ Ball Rolling ###############
    # n_videos = 20
    # save_dir = "videos/left_right"
    # os.makedirs(save_dir, exist_ok=True)

    # n_right = 0
    # n_left = 0

    # for i in range(n_videos):
    #     print(f"\nRecording video {i}...")
    #     # record video
    #     video_path = os.path.join(save_dir, f"video{i}.mp4")
    #     action_hist, eef_pos_hist, ball_pos_hist, ball_side = record_left_right(
    #         video_path=video_path,
    #         camera_names="frontview2",
    #         ball_rgba=(0.75,0.75,0.75,1),
    #     )
        
    #     if ball_side == "left":
    #         n_left += 1
    #     elif ball_side == "right":
    #         n_right += 1
    #     print(f"left: {n_left}, right: {n_right}")

    #     # make 2 copies of video and rename
    #     shutil.copyfile(video_path, os.path.join(save_dir, f"video{i}_goal_right_result_{ball_side}.mp4"))
    #     os.rename(video_path, os.path.join(save_dir, f"video{i}_goal_left_result_{ball_side}.mp4"))

    #     # save datafile
    #     n = len(action_hist)
    #     csv_path = os.path.join(save_dir, f"video{i}_goal_left_result_{ball_side}.csv")
    #     df = pd.DataFrame(
    #         data={
    #             "step" : np.arange(n),
    #             "action" : pd.Series(action_hist),
    #             "eef_pos" : pd.Series(eef_pos_hist),
    #             "ball_pos" : pd.Series(ball_pos_hist),
    #             "ball_ended_up_on" : pd.Series([ball_side] * n),
    #             "ball_goal_side" : pd.Series(["left"] * n)
    #         }
    #     )
    #     df.to_csv(csv_path, index=False)

    #     csv_path = os.path.join(save_dir, f"video{i}_goal_right_result_{ball_side}.csv")
    #     df = pd.DataFrame(
    #         data={
    #             "step" : np.arange(n),
    #             "action" : pd.Series(action_hist),
    #             "eef_pos" : pd.Series(eef_pos_hist),
    #             "ball_pos" : pd.Series(ball_pos_hist),
    #             "ball_ended_up_on" : pd.Series([ball_side] * n),
    #             "ball_goal_side" : pd.Series(["right"] * n)
    #         }
    #     )
    #     df.to_csv(csv_path, index=False)


    ############ Obstacle Avoidance (Manual Record) ###############
    # camera_names = ["frontview", "sideview"]
    # target_half_size=(0.05,0.05,0.001)
    # obstacle_half_size=(0.025, 0.025, 0.12)

    # # initialize an environment with offscreen renderer
    # env = make(
    #     env_name="Reaching2DObstacle",
    #     controller_configs=load_controller_config(default_controller="OSC_POSE"),
    #     robots="Panda",
    #     has_renderer=False,
    #     has_offscreen_renderer=True,
    #     render_camera="agentview",
    #     camera_names=camera_names,
    #     use_camera_obs=True,
    #     control_freq=20,
    #     ignore_done=True,
    #     target_half_size=target_half_size,
    #     obstacle_half_size=obstacle_half_size,
    #     random_init=True,
    #     random_target=True,
    #     camera_heights=512,
    #     camera_widths=512,
    # )
    # env = VisualizationWrapper(env, indicator_configs=None)
    
    # # initialize spacemouse
    # device = SpaceMouse(
    #         vendor_id=9583,
    #         product_id=50734,
    #         pos_sensitivity=1.0,
    #         rot_sensitivity=1.0,
    #     )

    # device.start_control()

    # n_videos = 5
    # save_dir = "videos/obstacle_avoidance"
    # os.makedirs(save_dir, exist_ok=True)
    # # global recording

    # for i in range(n_videos):

    #     obs = env.reset()
    #     target_pos = obs["target_pos"]
    #     obstacle_pos = obs["obstacle_pos"]

    #     # record video
    #     action_hist, eef_pos_hist, reward_hist = manual_record_spacemouse(
    #         env=env,
    #         device=device,
    #         video_path=os.path.join(save_dir, f"video{i}.mp4"),
    #         camera_names=camera_names
    #     )

    #     # record csv datafile
    #     n_steps = len(action_hist)
    #     good = np.all(np.array(reward_hist) >= 0)
    #     df = pd.DataFrame(
    #         data={
    #             "step" : np.arange(len(action_hist)),
    #             "action" : pd.Series(action_hist),
    #             "eef_pos" : pd.Series(eef_pos_hist),
    #             "target_pos" : [target_pos] * n_steps,
    #             "obstacle_pos" : [obstacle_pos] * n_steps,
    #             "target_half_size" : [target_half_size] * n_steps,
    #             "obstacle_half_size" : [obstacle_half_size] * n_steps,
    #             "good" : [good] * n_steps, # if collision occurs, it is a bad demo
    #         }
    #     )

    #     label = "good" if good else "bad"
    #     df.to_csv(os.path.join(save_dir, f"video{i}_{label}.csv"), index=False)

    # cv2.destroyAllWindows()
    # env.close()

    ############ Record Equal Length 2D Reaching ##############
    # save_path = "videos/2d_reaching"
    # good_path = os.path.join(save_path, "good")
    # bad_path = os.path.join(save_path, "bad")
    # os.makedirs(save_path, exist_ok=True)
    # os.makedirs(good_path, exist_ok=True)
    # os.makedirs(bad_path, exist_ok=True)

    # n_videos = 25
    # steps = 100

    # for i in range(n_videos):
    #     print(f"\nrecording good {i}")
    #     record_videos_2d_reaching(
    #         video_path=os.path.join(good_path, f"good{i}.mp4"),
    #         csv_path=os.path.join(good_path, f"good{i}.csv"),
    #         camera_names="frontview",
    #         good=True,
    #         timesteps=steps,
    #     )
    #     print(f"\nrecording bad {i}")
    #     record_videos_2d_reaching(
    #         video_path=os.path.join(bad_path, f"bad{i}.mp4"),
    #         csv_path=os.path.join(bad_path, f"bad{i}.csv"),
    #         camera_names="frontview",
    #         good=False,
    #         timesteps=steps,
    #     )

    ############## Record Drawer Stimulus ###############
    # views = ["sideview", "agentview"]
    # speeds = [0.1, 0.2, 0.3, 0.4]
    # save_dir = "videos/drawer_large"
    # os.makedirs(save_dir, exist_ok=True)
    
    # for view, speed in [(view, speed) for view in views for speed in speeds]:
    #     print(f"recording {view}, {speed}")
    #     record_video_drawer(
    #         video_path=os.path.join(save_dir, f"{view}_speed{speed}.mp4"),
    #         csv_path=os.path.join(save_dir, f"{view}_speed{speed}.csv"),
    #         camera_names=view,
    #         thresh=speed/100,
    #         speed=speed,
    #     )

    ############ Record Abstract State Stimulus ###############
    # views = ["agentview"]
    # obj_colors = ["red"]
    # states = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # # start_states = [1]
    # # end_states = [4, 6]
    # speed = 0.15
    # save_root_dir = "videos/abstract_states_color_grid"
    # os.makedirs(save_root_dir, exist_ok=True)

    # for view in views:
    #     for color in obj_colors:
    #         save_dir = os.path.join(save_root_dir, f"{view}_{color}")
    #         os.makedirs(save_dir, exist_ok=True)
    #         for start in states:
    #             for end in states:
    #                 if start == end:
    #                     continue
    #                 print(f"recording {view} {color} {start} to {end}")
    #                 # record_video_abstract_states(start_state=1, end_state=2)
    #                 record_video_abstract_states(
    #                     start_state=start,
    #                     end_state=end,
    #                     video_path=os.path.join(save_dir, f"state{start}_to_{end}.mp4"),
    #                     csv_path=os.path.join(save_dir, f"state{start}_to_{end}.csv"),
    #                     camera_names=view,
    #                     obj_rgba=colors[color],
    #                     speed=speed,
    #                 )

    ########## Record Lift ###############
    # n_videos = 15
    # obj_colors = ["blue", "gray", "red"]
    # save_root_dir = "videos/lift"
    # views = ["frontview", "sideview"]
    # for view in views:
    #     for color in obj_colors:
    #         good_dir = os.path.join(save_root_dir, f"{view}_{color}/good")
    #         bad_dir = os.path.join(save_root_dir, f"{view}_{color}/bad")
    #         os.makedirs(good_dir, exist_ok=True)
    #         os.makedirs(bad_dir, exist_ok=True)
    #         for i in range(n_videos):
    #             print(f"recording good {view} {color} {i}")
    #             record_videos_lift(
    #                 video_path=os.path.join(good_dir, f"lift_good_{i}.mp4"),
    #                 csv_path=os.path.join(good_dir, f"lift_good_{i}.csv"),
    #                 camera_names=view,
    #                 obj_rgba=colors[color],
    #                 good=True,
    #             )
    #             print(f"recording bad {view} {color} {i}")
    #             record_videos_lift(
    #                 video_path=os.path.join(bad_dir, f"lift_bad_{i}.mp4"),
    #                 csv_path=os.path.join(bad_dir, f"lift_bad_{i}.csv"),
    #                 camera_names=view,
    #                 obj_rgba=colors[color],
    #                 good=False,
    #             )

    ######### Record Cube Sliding Video ############
    # import time
    # cases = {
    #     "x_low2high": {"start_pos" : np.array([-0.25, 0, 0]), "end_pos" : np.array([0.25, 0, 0])},
    #     "x_high2low": {"start_pos" : np.array([0.25, 0, 0]), "end_pos" : np.array([-0.25, 0, 0])},
    #     "y_low2high": {"start_pos" : np.array([0, -0.35, 0]), "end_pos" : np.array([0, 0.35, 0])},
    #     "y_high2low": {"start_pos" : np.array([0, 0.35, 0]), "end_pos" : np.array([0, -0.35, 0])},
    #     "z_low2high": {"start_pos" : np.array([0, 0, 0]), "end_pos" : np.array([0, 0, 0.5])},
    #     "z_high2low": {"start_pos" : np.array([0, 0, 0.5]), "end_pos" : np.array([0, 0, 0])},
    # }

    # speeds = (1, 0.5, 0.25)
    # # views = ["frontview", "sideview"]
    # views = ["agentview2"]

    # obj_colors = ("red", "blue", "gray")

    # save_root_dir = "videos/cube_slide"
    # os.makedirs(save_root_dir, exist_ok=True)

    # for view in views:
    #     for color in obj_colors:
    #         save_dir = os.path.join(save_root_dir, f"{view}_{color}")
    #         os.makedirs(save_dir, exist_ok=True)
    #         for case in cases:
    #             for speed in speeds:
    #                 print(f"recording {view} {color} {case} speed {speed}")
    #                 start_time = time.time()
    #                 record_videos_move_obj(
    #                     initial_eef_pos=cases[case]["start_pos"],
    #                     final_eef_pos=cases[case]["end_pos"],
    #                     speed=speed,
    #                     video_path=os.path.join(save_dir, f"{case}_speed{speed}.mp4"),
    #                     csv_path=os.path.join(save_dir, f"{case}_speed{speed}.csv"),
    #                     camera_names=view,
    #                     obj_rgba=colors[color],
    #                 )
    #                 print("complete time ", time.time() - start_time)
    
    # record_videos_drop()
    
    # csv_path = "/home/ayanoh/robosuite/myscripts/videos/good_bad_mix/csvs"
    # csv_files = os.listdir(csv_path)
    # for file in csv_files:
    #     add_eef_in_target_to_csv(os.path.join(csv_path, file))

    # source_dir = "/home/ayanoh/robosuite/myscripts/videos/good_bad_mix"
    # save_dir = "/home/ayanoh/robosuite/myscripts/videos/trimmed_100steps"

    # csv_source_dir = os.path.join(source_dir, "csvs")
    # vid_source_dir = os.path.join(source_dir, "mp4s")
    # csv_files = os.listdir(csv_source_dir)
    # vid_files = os.listdir(vid_source_dir)
    # csv_files.sort()
    # vid_files.sort()

    # # pdb.set_trace()
    # if not len(csv_files) == len(vid_files):
    #     raise Exception("every video file must have corresponding csv file")

    # os.makedirs(save_dir, exist_ok=True)

    # for vid, csv in zip(vid_files, csv_files):
    #     print(f"============Vid {vid}, CSV {csv}================")

    #     vid_name = Path(vid).stem
    #     if not vid_name == Path(csv).stem:
    #         raise Exception("wrong vid-csv combination")

    #     # save_folder = os.path.join(save_dir, vid_name)
    #     # os.makedirs(save_folder)


    #     trim_video(
    #         vid_path=os.path.join(vid_source_dir, vid),
    #         csv_path=os.path.join(csv_source_dir, csv),
    #         save_folder=save_dir,
    #         use_prefix=True,
    #         len_thresh=100
    #     )        
