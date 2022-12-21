import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import GREEN, add_to_dict, array_to_string
import pdb

class TargetMarker3dObject(CompositeObject):
    """
    Generates a floating target marker with a square base (has collision) that sits on a surface
    and target (no collision) floating above the base

    Args:
        name (str): Name of this target object

        base_half_size (2-array of float): If specified, defines the base dimensions (side-lengths, thickness)

        target_half_size (n-array of float): If specified, defines the target dimensions
        
        target_height (float): distance from bottom surface of base to center of target

        density (float): Density value to use for all geoms. Defaults to 1000

        rgba (4-array or None): If specified, sets pot body rgba values
    """

    def __init__(
        self,
        name,
        base_half_size=(0.05, 0.005),
        target_half_size=(0.025, 0.05),
        target_height=0.2,
        density=1000,
        rgba=None,
    ):
        # Set name
        self._name = name

        # Set object attributes
        self.base_half_size = base_half_size
        self.target_half_size = target_half_size
        self.target_height = target_height
        self.density = density
        self.rgba = np.array(rgba) if rgba else GREEN

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
        body_radius = self.target_half_size[0]
        if self.base_half_size[0] > self.target_half_size[0]:
            body_radius = self.base_half_size[0]
        
        full_size = np.array(
            (
                body_radius,
                body_radius,
                # 0.5 * self.base_half_size[1] + self.target_height + self.target_half_size[1]
                self.base_half_size[1]
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

        # Base geom
        name = f"base"
        self.base = [name]
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0, 0, self.base_half_size[1]),
            geom_quats=(0, 0, 0, 1),
            geom_sizes=(self.base_half_size[0], self.base_half_size[0], self.base_half_size[1]),
            geom_names=name,
            geom_rgbas=self.rgba,
            geom_frictions=None,
            density=self.density,
        )
        # Add body site
        base_site = self.get_site_attrib_template()
        center_name = "center"
        base_site.update(
            {
                "name": center_name,
                "size": "0.005",
            }
        )
        site_attrs.append(base_site)
        self._important_sites["center"] = self.naming_prefix + center_name

        target_site = self.get_site_attrib_template()
        target_center_name = "target"
        target_site.update(
            {
                "name": target_center_name,
                "pos": "0 0 " + str(self.target_height - self.base_half_size[1]),
                "size": array_to_string(self.target_half_size),
                "type": "cylinder",
                "rgba": "0 1 0 0.3",
                "group": "1",
            }
        )
        site_attrs.append(target_site)
        self._important_sites[target_center_name] = self.naming_prefix + target_center_name

        obj_args.update(base_args)
        obj_args["sites"] = site_attrs

        # Return this dict
        return obj_args

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle0'`: Name of handle0 location site
                :`'handle1'`: Name of handle1 location site
        """
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
