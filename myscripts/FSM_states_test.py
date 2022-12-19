import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.primitive_skills import PrimitiveSkill
from robosuite.utils.state_machine import StateMachine
import pdb


# test FSM state for drop2
def pick_block_transition(env, obs, **kwargs):
    """
    pick_block state for drop2 task
    """
    
    # call primitive
    prim = PrimitiveSkill(env)
    obs, reward, done, info = prim.pick(
        obs=obs,
        goal_pos=kwargs.get("goal_pos"),
        obj_id=kwargs.get("obj_id"),
        waypoint_height=kwargs.get("waypoint_height"),        
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
        params = {
            "obj_id" : env.stage_body_id,
        }
    else:
        # if cube was not picked up, try again
        next_state = "pick_block"
        params = {
            "obj_id" : env.block_body_id,
        }

    return obs, next_state

def drop_cube_transition(env, obs, **kwargs):
    """
    drop_cube state for drop2 task
    """

    # call primitive
    prim = PrimitiveSkill(env)
    obs, reward, done, info = prim.place(
        obs=obs,
        goal_pos=kwargs.get("goal_pos"),
        obj_id=kwargs.get("obj_id"),
        waypoint_height=kwargs.get("waypoint_height"),
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
        next_state = "pick_cube"
        params = {
           "obj_id" : env.cube_body_id,
        }

    return obs, next_state, params

def complete_state(env, obs, **kwargs):
    """
    complete state after task is complete
    """
    print("----------TASK COMPLETED----------")


env = suite.make(
    env_name="Drop2",
    controller_configs=load_controller_config(default_controller="OSC_POSE"),
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    render_camera="frontview",
    use_camera_obs=False,
    control_freq=20,
    ignore_done=True,
)
sm = StateMachine(env)
