import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import RED, add_to_dict


class BlockWithLEDObject(CompositeObject):
    """
    Generates a block with LED-like spheres attached

    Args:
        name (str): Name of this object

        body_half_size (3-array of float): If specified, defines the (x,y,z) half-dimensions of the main block
            body. Otherwise, defaults to [0.07, 0.07, 0.07]

        led_radius (float): radius of LED
        
        density (float): Density value to use for all geoms. Defaults to 1000

        block_rgba (4-array of floats or None): If specified, sets pot body rgba values

        led_rgba (4-array of floats None)

    """

    def __init__(
        self,
        name,
        body_half_size=(0.025, 0.025, 0.025),
        led_radius=0.005,
        density=1000,
        block_rgba=None,
        led_rgba=None,
        frictions=(1.0, 0.005, 0.0001),
    ):
        # Set name
        self._name = name

        # Set object attributes
        self.body_half_size = np.array(body_half_size)
        self.density = density
        self.led_radius = led_radius
        self.block_rgba = np.array(block_rgba) if block_rgba else np.array([1, 1, 1, 1]) # defaults to white
        self.led_rgba = np.array(led_rgba) if led_rgba else np.array([1, 0, 0, 1]) # defaults to red

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
            geom_types="box",
            geom_locations=(0, 0, 0),
            geom_quats=(0, 0, 0, 1),
            geom_sizes=self.body_half_size,
            geom_names=name,
            geom_rgbas=self.block_rgba,
            geom_frictions=self.frictions,
            density=self.density,
        )

        # LED1
        name = "led1"
        self.led1 = [name]
        add_to_dict(
            dic=obj_args,
            geom_types="sphere",
            geom_locations=(self.body_half_size[0], 0, 0),
            geom_quats=(0, 0, 0, 1),
            geom_sizes=np.array([self.led_radius]),
            geom_names=name,
            geom_rgbas=self.led_rgba,
            geom_frictions=self.frictions,
            density=self.density,
        )

        # LED2
        name = "led2"
        self.led2 = [name]
        add_to_dict(
            dic=obj_args,
            geom_types="sphere",
            geom_locations=(-self.body_half_size[0], 0, 0),
            geom_quats=(0, 0, 0, 1),
            geom_sizes=np.array([self.led_radius]),
            geom_names=name,
            geom_rgbas=self.led_rgba,
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

        # # Add LED sites
        # led_site1 = self.get_site_attrib_template()
        # led1_name = "led1"
        # led_site1.update(
        #     {
        #         "name" : led1_name,
        #         "pos" : str(self.body_half_size[0]) + " 0.0 0.0",
        #         "size" : str(self.led_radius),
        #         "type" : "sphere",
        #         "rgba" : " ".join(map(str, self.led_rgba)),
        #         "group" : "1",
        #     }
        # )
        # site_attrs.append(led_site1)
        # self._important_sites[led1_name] = self.naming_prefix + led1_name

        # led_site2 = self.get_site_attrib_template()
        # led2_name = "led2"
        # led_site2.update(
        #     {
        #         "name" : led2_name,
        #         "pos" : str(-self.body_half_size[0]) + " 0.0 0.0",
        #         "size" : str(self.led_radius),
        #         "type" : "sphere",
        #         "rgba" : " ".join(map(str, self.led_rgba)),
        #         "group" : "1",
        #     }
        # )
        # site_attrs.append(led_site2)
        # self._important_sites[led2_name] = self.naming_prefix + led2_name

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
