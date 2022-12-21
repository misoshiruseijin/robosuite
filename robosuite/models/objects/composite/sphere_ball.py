import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import RED, add_to_dict


class SphereBallObject(CompositeObject):
    """
    Generates a ball object

    Args:
        name (str): Name of this SphereBall
        
        size (1-tuple of float): radius of sphere
    """

    def __init__(
        self,
        name,
        size=0.025,
        density=1000,
        rgba=None,
        frictions=(1.0, 0.005, 0.0001),
    ):
        # Set name
        self._name = name

        # Set object attributes
        self.body_half_size = size
        self.density = density
        self.rgba = np.array(rgba) if rgba else RED
        self.frictions = frictions

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

        # main geom
        name = f"body"
        self.body = [name]
        add_to_dict(
            dic=obj_args,
            geom_types="sphere",
            geom_locations=(0, 0, 0),
            geom_quats=(0, 0, 0, 1),
            geom_sizes=self.body_half_size,
            geom_names=name,
            geom_rgbas=self.rgba,
            geom_frictions=self.frictions,
            density=self.density,
        )

        # Add body site
        stage_site = self.get_site_attrib_template()
        center_name = "center"
        stage_site.update(
            {
                "name": center_name,
                "size": "0.005",
            }
        )
        site_attrs.append(stage_site)
        # Add to important sites
        self._important_sites["center"] = self.naming_prefix + center_name

        # Add back in base args and site args
        obj_args.update(base_args)
        obj_args["sites"] = site_attrs  # All sites are part of main (top) body

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
