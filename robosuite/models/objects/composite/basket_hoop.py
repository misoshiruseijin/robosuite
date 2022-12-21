import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import RED, add_to_dict


class BasketHoopObject(CompositeObject):
    """
    Generates the rectangular basket object

    Args:
        name (str): Name of this Pot object

        body_half_size (3-array of float): If specified, defines the (x,y,z) half-dimensions of the main pot
            body. Otherwise, defaults to [0.07, 0.07, 0.07]
        
        density (float): Density value to use for all geoms. Defaults to 1000

        rgba (4-array or None): If specified, sets pot body rgba values

        thickness (float): How thick to make the pot body walls
    """

    def __init__(
        self,
        name,
        body_half_size=(0.25, 0.15, 0.125),
        density=1000,
        rgba=None,
        thickness=0.01,  # thickness of hoop and base
    ):
        # Set name
        self._name = name

        # Set object attributes
        self.body_half_size = np.array(body_half_size)
        self.thickness = thickness
        self.density = density
        self.rgba = np.array(rgba) if rgba else RED
        self.pole_radius = 0.005
        self.pole_half_height = self.body_half_size[2] - self.thickness
        self.hoop_half_size = np.array([0.1, 0.1, self.thickness/2])
        self.hoop_width = 0.01 # difference between outer and inner edges
        # self.overhang_half_size = np.array([self.body_half_size[0] - self.hoop_half_size[0], self.pole_radius, self.pole_radius])
        self.overhang_half_size = np.array([0.075, self.pole_radius, self.pole_radius])

        # Element references to be filled when generated
        self.hoop_base = None

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

        # Initialize geom lists

        # hoop
        name = "side1"
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0, self.hoop_half_size[1] - self.hoop_width/2, self.body_half_size[2] - self.hoop_half_size[2]),
            geom_quats=(0, 0, 0, 1),
            geom_sizes=np.array([self.hoop_half_size[1], self.hoop_width/2, self.hoop_half_size[2]]),
            geom_names=name,
            geom_rgbas=self.rgba,
            geom_frictions=None,
            density=self.density,
        )

        name = "side2"
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0, -self.hoop_half_size[1] + self.hoop_width/2, self.body_half_size[2] - self.hoop_half_size[2]),
            geom_quats=(0, 0, 0, 1),
            geom_sizes=np.array([self.hoop_half_size[0], self.hoop_width/2, self.hoop_half_size[2]]),
            geom_names=name,
            geom_rgbas=self.rgba,
            geom_frictions=None,
            density=self.density,
        )

        name = "side3"
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(-self.hoop_half_size[0] + self.hoop_width/2, 0, self.body_half_size[2] - self.hoop_half_size[2]),
            geom_quats=(0, 0, 0, 1),
            geom_sizes=np.array([self.hoop_width/2, self.hoop_half_size[1], self.hoop_half_size[2]]),
            geom_names=name,
            geom_rgbas=self.rgba,
            geom_frictions=None,
            density=self.density,
        )

        name = "side4"
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(self.hoop_half_size[0] - self.hoop_width/2, 0, self.body_half_size[2] - self.hoop_half_size[2]),
            geom_quats=(0, 0, 0, 1),
            geom_sizes=np.array([self.hoop_width/2, self.hoop_half_size[1], self.hoop_half_size[2]]),
            geom_names=name,
            geom_rgbas=self.rgba,
            geom_frictions=None,
            density=self.density,
        )

        # overhang
        name = "overhang"
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(-self.hoop_half_size[0] - self.overhang_half_size[0], 0, self.body_half_size[2] - self.overhang_half_size[2]),
            geom_quats=(0, 0, 0, 1),
            geom_sizes=self.overhang_half_size,
            geom_names=name,
            geom_rgbas=self.rgba,
            geom_frictions=None,
            density=self.density,
        )

        # Pole
        name = f"pole"
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(-self.hoop_half_size[0] - 2*self.overhang_half_size[0] + self.pole_radius, 0, 0),
            geom_quats=(0, 0, 0, 1),
            geom_sizes=np.array([self.pole_radius, self.pole_radius, self.pole_half_height]),
            geom_names=name,
            geom_rgbas=self.rgba,
            geom_frictions=None,
            density=self.density,
        )

        # Base geom
        name = f"base"
        self.hoop_base = [name]
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0, 0, -self.body_half_size[2] + self.thickness / 2),
            geom_quats=(0, 0, 0, 1),
            geom_sizes=np.array([self.body_half_size[0], self.body_half_size[1], self.thickness/2]),
            geom_names=name,
            geom_rgbas=(255, 255, 255, 1),
            geom_frictions=[100,100,100],
            density=self.density,
        )

        
        # Add hoop body site
        hoop_site = self.get_site_attrib_template()
        center_name = "center"
        hoop_site.update(
            {
                "name": center_name,
                "size": "0.005",
            }
        )
        site_attrs.append(hoop_site)
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
