from robosuite.environments.base import make

# Manipulation environments
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.stack import Stack
from robosuite.environments.manipulation.nut_assembly import NutAssembly
from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite.environments.manipulation.door import Door
from robosuite.environments.manipulation.wipe import Wipe
from robosuite.environments.manipulation.two_arm_lift import TwoArmLift
from robosuite.environments.manipulation.two_arm_peg_in_hole import TwoArmPegInHole
from robosuite.environments.manipulation.two_arm_handover import TwoArmHandover
from robosuite.environments.manipulation.ball_basket import BallBasket
from robosuite.environments.manipulation.reaching_saywerEF import ReachingSawyerEF
from robosuite.environments.manipulation.reaching_frankaBC import ReachingFrankaBC
from robosuite.environments.manipulation.reaching_2d import Reaching2D
from robosuite.environments.manipulation.drop import Drop
from robosuite.environments.manipulation.object_and_table import Object_and_Table
from robosuite.environments.manipulation.lift2 import Lift2
from robosuite.environments.manipulation.lift_flash import LiftFlash
from robosuite.environments.manipulation.grid_wall import GridWall
from robosuite.environments.manipulation.hammer_place import HammerPlaceEnv
from robosuite.environments.manipulation.drawer import DrawerEnv
from robosuite.environments.manipulation.reaching_2d_obstacle import Reaching2DObstacle
from robosuite.environments.manipulation.pick_place_primitive import PickPlacePrimitive
from robosuite.environments.manipulation.left_right import LeftRight
from robosuite.environments.manipulation.drop2 import Drop2
from robosuite.environments.manipulation.stack_custom import StackCustom
from robosuite.environments.manipulation.cleanup import Cleanup
from robosuite.environments.manipulation.POC_reaching import POCReaching
from robosuite.environments.manipulation.stack_maple import StackMAPLE

from robosuite.environments import ALL_ENVIRONMENTS
from robosuite.controllers import ALL_CONTROLLERS, load_controller_config
from robosuite.robots import ALL_ROBOTS
from robosuite.models.grippers import ALL_GRIPPERS

__version__ = "1.3.1"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\  [~]\/    |//  |
     ] [   OOO      /o|__|
"""
