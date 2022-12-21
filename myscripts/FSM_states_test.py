import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.primitive_skills import PrimitiveSkill
from robosuite.utils.state_machine import StateMachine
import numpy as np
import pdb


# test FSM state for drop2
def pick_block_transition(env, obs, params):
    """
    pick_block state for drop2 task
    """

    print("PICK BLOCK STATE")
    # call primitive
    prim = PrimitiveSkill(env)
    obs, reward, done, info = prim.pick(
        obs=obs,
        goal_pos=params.get("goal_pos"),
        obj_id=params.get("obj_id"),
        waypoint_height=params.get("waypoint_height"),        
    )

    # check success (grasping cube)
    success = env._check_grasp(
        gripper=env.robots[0].gripper,
        object_geoms=env.block
    )
    params = {}

    # state transition
    print("Pick block success: ", success)
    if success:
        # if cube is successfully picked up, advance to drop phase
        next_state = "drop_block"
        drop_pos = env.sim.data.body_xpos[env.stage_body_id]
        drop_pos = np.array([drop_pos[0], drop_pos[1], drop_pos[2] + env.block.body_half_size[2]+0.05])
        params = {
            # "obj_id" : env.stage_body_id,
            "goal_pos" : drop_pos,
        }
    else:
        # if cube was not picked up, try again
        next_state = "pick_block"
        params = {
            "obj_id" : env.block_body_id,
        }

    return obs, next_state, params

def drop_block_transition(env, obs, params):
    """
    drop_block state for drop2 task
    """

    print("DROP BLOCK STATE")
    # call primitive
    prim = PrimitiveSkill(env)
    obs, reward, done, info = prim.place(
        obs=obs,
        goal_pos=params.get("goal_pos"),
        obj_id=params.get("obj_id"),
        waypoint_height=params.get("waypoint_height"),
    )

    # check success
    success = env._check_success()

    params = {} # parameters for next primitive

    # state transition
    if success:
        # cube ended up on stage - go to idle
        next_state = "complete"
    else:
        # cube was dropped elsewhere - pick it up again
        next_state = "pick_block"
        params = {
           "obj_id" : env.block_body_id,
        }

    return obs, next_state, params

def complete_state_transition(env, obs, params={}):
    """
    complete state after task is complete
    """
    print("----------TASK COMPLETED----------")
    return None


env = suite.make(
    env_name="Drop2",
    controller_configs=load_controller_config(default_controller="OSC_POSITION"),
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    render_camera="frontview",
    use_camera_obs=False,
    control_freq=20,
    ignore_done=True,
)

state_transitions = {
    "pick_block" : pick_block_transition,
    "drop_block" : drop_block_transition,
    "complete" : complete_state_transition,    
}
initial_params = {
    "obj_id" : env.block_body_id,
}

sm = StateMachine(
    env=env,
    states=state_transitions,
    params=initial_params,
)
sm.set_initial_state("pick_block")
sm.set_end_state("complete")
sm.run()