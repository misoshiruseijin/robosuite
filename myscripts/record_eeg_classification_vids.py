"""
Modified version of robosuite.demos.demo_video_recording

Record video of agent episodes with the imageio library.
This script uses offscreen rendering.

Example:
    $ python demo_video_recording.py --environment Lift --robots Panda
"""

import imageio
import numpy as np

import robosuite.utils.macros as macros
from robosuite import make
from robosuite.wrappers.visualization_wrapper import VisualizationWrapper
from robosuite.controllers import load_controller_config

import pandas as pd
import os
from pathlib import Path

import pdb
# Set the image convention to opencv so that the images are automatically rendered "right side up" when using imageio
# (which uses opencv convention)
macros.IMAGE_CONVENTION = "opencv"

np.random.seed(1)

################## Franka 2D Reaching ###########################
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

def record_videos():
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

def record_videos_lift(video_path="video.mp4", csv_path="data.csv", camera_names="frontview", obj_rgba=(1,0,0,1)):

    def generate_action_lift(target_pos, closed_gripper, noise_strength=0.02):
        """
        Args:
            target_pos: good position to pick object up
            eef_pos: current eef position

        Returns:
            action
        """
        thresh = 0.02
        action_scale = 0.5
        goal_pos = target_pos + noise_strength * np.random.uniform(-1, 1, 3)
        closed = closed_gripper

        u = (goal_pos - eef_pos) / np.linalg.norm(goal_pos - eef_pos)

        if closed_gripper:
            # lift object up
            action = np.array([0, 0, 0.25, 0, 0, 0, 1])

        elif np.all(np.abs(goal_pos - eef_pos) < thresh):
            # close gripper if eef is at goal position
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

    done = False
    closed = False

    n_frames = 0
    lift_steps = 0

    while not done and lift_steps < 30: # done checks for success, lift_steps make sure recording stops for failure case
        if n_frames % 100 == 0:
            print("frame ", n_frames)
        
        action, closed = generate_action_lift(obj_pos, closed, noise_strength=0.05)
        obs, reward, done, info = env.step(action)
        frame = obs[camera_names + "_image"]
        writer.append_data(frame)

        eef_pos = obs["robot0_eef_pos"]
        eef_pos_hist.append(eef_pos)
        action_hist.append(action)
        obj_pos_hist.append(obs["cube_pos"])
        success_hist.append(done)

        # print("action ", action)
        print("eef pos ", eef_pos)
        # print("error ", eef_pos - final_eef_pos)
        n_frames += 1
        lift_steps += closed

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


if __name__ == "__main__":

    ########## Sliding Cube in All Directions ###############
    record_videos_lift()

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
    # colors = {
    #     "red" : (1,0,0,1),
    #     "green" : (0,1,0,1),
    #     "blue" : (0,0,1,1),
    #     "yellow" : (1,1,0,1),
    #     "gray" : (0.75,0.75,0.75,1),
    # }

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
