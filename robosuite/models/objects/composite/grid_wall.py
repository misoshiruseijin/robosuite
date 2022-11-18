import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import RED, add_to_dict
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import RED, add_to_dict


# class GridWallObject(CompositeObject):
#     """
#     Generates a stage object (rectangular table with legs)

#     Args:
#         name (str): Name of this Stage object

#         body_half_size (3-array of float): If specified, defines the (x,y,z) half-dimensions of the main pot
#             body. Otherwise, defaults to [0.07, 0.07, 0.07]
        
#         density (float): Density value to use for all geoms. Defaults to 1000

#         rgba (4-array or None): If specified, sets pot body rgba values

#         thickness (float): How thick to make the pot body walls
#     """

#     def __init__(
#         self,
#         name,
#         body_half_size=(0.20, 0.20, 0.05),
#         target_half_size=(0.08, 0.15),
#         density=1000,
#         line_width=0.05,
#         rgba=None,
#     ):
#         # Set name
#         self._name = name

#         # Set object attributes
#         self.body_half_size = np.array(body_half_size)
#         self.target_half_size = np.array(target_half_size)
#         self.thickness = body_half_size[2]
#         self.target_position = (0, 0) # TODO - make this editable parameter
#         self.density = density
#         self.rgba = np.array(rgba) if rgba else RED
#         self.line_width = line_width

#         # Element references to be filled when generated
#         self.top_surface = None

#         # Other private attributes
#         self._important_sites = {}

#         # Create dictionary of values to create geoms for composite object and run super init
#         super().__init__(**self._get_geom_attrs())

#     def _get_geom_attrs(self):
#         """
#         Creates geom elements that will be passed to superclass CompositeObject constructor

#         Returns:
#             dict: args to be used by CompositeObject to generate geoms
#         """
#         full_size = np.array(
#             (
#                 self.body_half_size,
#                 self.body_half_size,
#                 self.body_half_size,
#             )
#         )
#         # Initialize dict of obj args that we'll pass to the CompositeObject constructor
#         base_args = {
#             "total_size": full_size / 2.0,
#             "name": self.name,
#             "locations_relative_to_center": True,
#             "obj_types": "all",
#         }
#         site_attrs = []
#         obj_args = {}

#         # Walls 
#         name = f"wall1"
#         self.wall1 = [name]
#         add_to_dict(
#             dic=obj_args,
#             geom_types="box",
#             geom_locations=(0, self.target_position[1] - self.target_half_size[1], self.thickness / 2),
#             geom_quats=(0, 0, 0, 1),
#             geom_sizes=np.array([self.body_half_size[0]*2, self.line_width, self.thickness]),
#             geom_names=name,
#             geom_rgbas=(1, 0, 0, 1),
#             geom_frictions=None,
#             density=self.density,
#         )

#         name = f"wall2"
#         self.wall1 = [name]
#         add_to_dict(
#             dic=obj_args,
#             geom_types="box",
#             geom_locations=(0, self.target_position[1] + self.target_half_size[1], self.thickness / 2),
#             geom_quats=(0, 0, 0, 1),
#             geom_sizes=np.array([self.body_half_size[0]*2, self.line_width, self.thickness]),
#             geom_names=name,
#             geom_rgbas=(0, 1, 0.5, 1),
#             geom_frictions=None,
#             density=self.density,
#         )

#         name = f"wall3"
#         self.wall1 = [name]
#         add_to_dict(
#             dic=obj_args,
#             geom_types="box",
#             geom_locations=(self.target_position[0] + self.target_half_size[0], 0, self.thickness / 2),
#             geom_quats=(0, 0, 0, 1),
#             geom_sizes=np.array([self.line_width, self.body_half_size[1]*2, self.thickness]),
#             geom_names=name,
#             geom_rgbas=(0.1, 0.5, 1, 1),
#             geom_frictions=None,
#             density=self.density,
#         )

#         name = f"wall4"
#         self.wall1 = [name]
#         add_to_dict(
#             dic=obj_args,
#             geom_types="box",
#             geom_locations=(self.target_position[0] - self.target_half_size[0], 0, self.thickness / 2),
#             geom_quats=(0, 0, 0, 1),
#             geom_sizes=np.array([self.line_width, self.body_half_size[1]*2, self.thickness]),
#             geom_names=name,
#             geom_rgbas=(0, 0, 0, 1),
#             geom_frictions=None,
#             density=self.density,
#         )

        
#         # Add body site
#         grid_site = self.get_site_attrib_template()
#         center_name = "center"
#         grid_site.update(
#             {
#                 "name": center_name,
#                 "size": "0.005",
#             }
#         )
#         site_attrs.append(grid_site)
#         # Add to important sites
#         self._important_sites["center"] = self.naming_prefix + center_name

#         # Add back in base args and site args
#         obj_args.update(base_args)
#         obj_args["sites"] = site_attrs  # All sites are part of main (top) body

#         # Return this dict
#         return obj_args

#     @property
#     def important_sites(self):
#         """
#         Returns:
#             dict: In addition to any default sites for this object, also provides the following entries

#                 :`'handle0'`: Name of handle0 location site
#                 :`'handle1'`: Name of handle1 location site
#         """
#         # Get dict from super call and add to it
#         dic = super().important_sites
#         dic.update(self._important_sites)
#         return dic

#     @property
#     def bottom_offset(self):
#         return np.array([0, 0, -1 * self.body_half_size[2]])

#     @property
#     def top_offset(self):
#         return np.array([0, 0, self.body_half_size[2]])

#     @property
#     def horizontal_radius(self):
#         return np.sqrt(2) * (max(self.body_half_size))


class GridWallObject(CompositeObject):
    """
    Generates a stage object (rectangular table with legs)

    Args:
        name (str): Name of this Stage object

        body_half_size (3-array of float): If specified, defines the (x,y,z) half-dimensions of the grid box
            body. Otherwise, defaults to [0.07, 0.07, 0.07]
        
        density (float): Density value to use for all geoms. Defaults to 1000

        rgba (4-array or None): If specified, sets pot body rgba values

        thickness (float): How thick to make the pot body walls
    """

    def __init__(
        self,
        name,
        body_half_size=(0.20, 0.20, 0.05),
        density=1000,
        wall_thickness=0.01,
        rgba=None,
    ):
        # Set name
        self._name = name

        # Set object attributes
        self.body_half_size = np.array(body_half_size)
        self.wall_height = body_half_size[2]
        self.wall_thickness = wall_thickness
        self.density = density
        self.rgba = np.array(rgba) if rgba else RED

        # wall positions
        self.wall_x = [
            -self.body_half_size[0]/3,
            self.body_half_size[0]/3,
        ]
        self.wall_y = [
            -self.body_half_size[1]/3,
            self.body_half_size[1]/3,
        ]

        # Element references to be filled when generated
        self.top_surface = None

        # Other private attributes
        self._important_sites = {}

        # Create dictionary of values to create geoms for composite object and run super init
        super().__init__(**self._get_geom_attrs())

    def _get_geom_attrs(self):
        """
        Creates geom elements that will be passed to superclass CompositeObject constructor

        Returns:
            dict: args to be used by CompositeObject to generate geoms
        """
        full_size = np.array(
            (
                self.body_half_size,
                self.body_half_size,
                self.body_half_size,
            )
        )
        # Initialize dict of obj args that we'll pass to the CompositeObject constructor
        base_args = {
            "total_size": full_size / 2.0,
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
        }
        site_attrs = []
        obj_args = {}

        # Walls 
        name = f"wall1"
        self.wall1 = [name]
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0, self.wall_y[0], self.wall_height/2),
            geom_quats=(0, 0, 0, 1),
            geom_sizes=np.array([self.body_half_size[0], self.wall_thickness/2, self.wall_height/2]),
            geom_names=name,
            geom_rgbas=self.rgba,
            geom_frictions=None,
            density=self.density,
        )

        name = f"wall2"
        self.wall2 = [name]
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0, self.wall_y[1], self.wall_height/2),
            geom_quats=(0, 0, 0, 1),
            geom_sizes=np.array([self.body_half_size[0], self.wall_thickness/2, self.wall_height/2]),
            geom_names=name,
            geom_rgbas=self.rgba,
            geom_frictions=None,
            density=self.density,
        )

        name = f"wall3"
        self.wall3 = [name]
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(self.wall_x[0], 0, self.wall_height/2),
            geom_quats=(0, 0, 0, 1),
            geom_sizes=np.array([self.wall_thickness/2, self.body_half_size[1], self.wall_height/2]),
            geom_names=name,
            geom_rgbas=self.rgba,
            geom_frictions=None,
            density=self.density,
        )
        
        name = f"wall4"
        self.wall4 = [name]
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(self.wall_x[1], 0, self.wall_height/2),
            geom_quats=(0, 0, 0, 1),
            geom_sizes=np.array([self.wall_thickness/2, self.body_half_size[1], self.wall_height/2]),
            geom_names=name,
            geom_rgbas=self.rgba,
            geom_frictions=None,
            density=self.density,
        )

        # Add body site
        grid_site = self.get_site_attrib_template()
        center_name = "center"
        grid_site.update(
            {
                "name": center_name,
                "size": "0.005",
            }
        )
        site_attrs.append(grid_site)
        # Add to important sites
        self._important_sites["center"] = self.naming_prefix + center_name

        # Add back in base args and site args
        obj_args.update(base_args)
        obj_args["sites"] = site_attrs  # All sites are part of main (top) body

        # Return this dict
        return obj_args

    @property
    def important_sites(self):
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update(self._important_sites)
        return dic

    @property
    def bottom_offset(self):
        return np.array([0, 0, -1 * self.body_half_size[2]])

    @property
    def top_offset(self):
        return np.array([0, 0, self.body_half_size[2]])

    @property
    def horizontal_radius(self):
        return np.sqrt(2) * (max(self.body_half_size))

    @property
    def wall_pos_x(self):
        return self.wall_x
    
    @property
    def wall_pos_y(self):
        return self.wall_y
