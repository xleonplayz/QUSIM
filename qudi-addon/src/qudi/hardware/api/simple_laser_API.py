# -*- coding: utf-8 -*-

"""
This module acts like a laser.

Copyright (c) 2021, the qudi developers. See the AUTHORS.md file at the top-level directory of this
distribution and on <https://github.com/Ulm-IQO/qudi-iqo-modules/>

This file is part of qudi.

Qudi is free software: you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

Qudi is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with qudi.
If not, see <https://www.gnu.org/licenses/>.
"""

import requests

from typing import List, FrozenSet, Dict
import logging


from qudi.interface.simple_laser_interface import SimpleLaserInterface
from qudi.interface.simple_laser_interface import LaserState, ShutterState, ControlMode


class SimpleLaserAPI(SimpleLaserInterface):
    """ Laser API for a dummy usage.

    Example config for copy-paste:

    laser_API:
        module.Class: 'laser.simple_laser_API.SimpleLaserAPI'
    """

    def __init__(self, **kwargs):
        """ """
        super().__init__(**kwargs)

        self._init_logger()
        self.log.debug("Initializing SimpleLaserAPI")

        self.api_url = 'http://localhost:8000/laser'
        requests.post(f"{self.api_url}/init")


    def _init_logger(self):
        """ Initialize logger for this module. 
            Default level is DEBUG.
        """
        if not hasattr(self, 'log'):
            self.log = logging.getLogger("qudi.hardware.api.simple_laser_API")
            self.log.setLevel(logging.DEBUG)

            ch = logging.StreamHandler()
            formatter = logging.Formatter('[FastCounter] %(asctime)s [%(levelname)s]: %(message)s')
            ch.setFormatter(formatter)
            self.log.addHandler(ch)

    def on_activate(self):
        """ Activate module. """
        self.log.debug("Activating SimpleLaserAPI")
        try:
            requests.post(f"{self.api_url}/on_activate")
        except Exception as e:
            self.log.error(f"Error during activation: {e}")

    def on_deactivate(self):
        """ Deactivate module. """
        self.log.debug("Deactivating SimpleLaserAPI")
        try:
            requests.get(f"{self.api_url}/on_deactivate")
        except Exception as e:
            self.log.error(f"Error during deactivation: {e}")

    def get_power_range(self) -> List[float]:
        """ Return power range in watts.

        @return float[2]: power range (min, max)
        """
        self.log.debug("Getting power range")
        try:
            response: List[float] = requests.get(f"{self.api_url}/power/range").json()
            return response
        except Exception as e:
            self.log.error(f"Error getting power range: {e}")
            return [0.0, 0.0]

    def get_power(self) -> float:
        """ Return actual aser power.

        @return float: Laser power in watts
        """
        self.log.debug("Getting actual laser power")
        try:
            response: float = requests.get(f"{self.api_url}/power").json()
            return response
        except Exception as e:
            self.log.error(f"Error getting power: {e}")
            return 0.0

    def get_power_setpoint(self) -> float:
        """ Return power setpoint.

        @return float: power setpoint in watts
        """
        self.log.debug("Getting power setpoint")
        try:
            response: float = requests.get(f"{self.api_url}/power/setpoint").json()
            return response
        except Exception as e:
            self.log.error(f"Error getting power setpoint: {e}")
            return 0.0

    def set_power(self, power: float):
        """ Set power setpoint.

        @param float power: power to set
        """
        self.log.debug(f"Setting power setpoint to {power} W")
        try:
            requests.post(f"{self.api_url}/power/setpoint?power_setpoint={power}")
        except Exception as e:
            self.log.error(f"Error setting power setpoint: {e}")

    def get_current_unit(self) -> str:
        """ Get unit for laser current.

        @return str: unit
        """
        self.log.debug("Getting current unit")
        try:
            response: str = requests.get(f"{self.api_url}/current/unit").json()
            return response
        except Exception as e:
            self.log.error(f"Error getting current unit: {e}")
            return "%"

    def get_current_range(self) -> List[float]:
        """ Get laser current range.

        @return float[2]: laser current range
        """
        self.log.debug("Getting current range")
        try:
            response: List[float] = requests.get(f"{self.api_url}/current/range").json()
            return response
        except Exception as e:
            self.log.error(f"Error getting current range: {e}")
            return [0.0, 0.0]

    def get_current(self) -> float:
        """ Get actual laser current

        @return float: laser current in current units
        """
        self.log.debug("Getting actual laser current")
        try:
            response: float = requests.get(f"{self.api_url}/current").json()
            return response
        except Exception as e:
            self.log.error(f"Error getting power: {e}")
            return 0.0

    def get_current_setpoint(self) -> float:
        """ Get laser current setpoint

        @return float: laser current setpoint
        """
        self.log.debug("Getting current setpoint")
        try:
            response: List[float] = requests.get(f"{self.api_url}/current/setpoint").json()
            return response
        except Exception as e:
            self.log.error(f"Error getting current setpoint: {e}")
            return 0.0

    def set_current(self, current: float):
        """ Set laser current setpoint

        @param float current: desired laser current setpoint
        """
        self.log.debug(f"Setting current setpoint to {current} {self.get_current_unit()}")
        try:
            requests.post(f"{self.api_url}/current/setpoint?current_setpoint={current}")
        except Exception as e:
            self.log.error(f"Error setting current setpoint: {e}")

    def allowed_control_modes(self) -> FrozenSet[ControlMode]:
        """ Get supported control modes

        @return frozenset: set of supported ControlMode enums
        """
        self.log.debug("Getting allowed control modes")
        try:
            response: FrozenSet[ControlMode] = requests.get(f"{self.api_url}/control_mode/allowed").json()
            return response
        except Exception as e:
            self.log.error(f"Error getting allowed control modes: {e}")
            return frozenset({ControlMode.POWER, ControlMode.CURRENT})

    def get_control_mode(self) -> ControlMode:
        """ Get the currently active control mode

        @return ControlMode: active control mode enum
        """
        self.log.debug("Getting current control mode")
        try:
            response: ControlMode = requests.get(f"{self.api_url}/control_mode").json()
            return response
        except Exception as e:
            self.log.error(f"Error getting control mode: {e}")
            return ControlMode.UNKNOWN

    def set_control_mode(self, control_mode: ControlMode):
        """ Set the active control mode

        @param ControlMode control_mode: desired control mode enum
        """
        self.log.debug(f"Setting control mode to {control_mode}")
        try:
            requests.post(f"{self.api_url}/control_mode?control_mode={control_mode}")
        except Exception as e:
            self.log.error(f"Error setting control mode: {e}")

    def on(self) -> LaserState:
        """ Turn on laser.

            @return LaserState: actual laser state
        """
        self.log.debug("Turning on laser")
        try:
            response: LaserState = requests.post(f"{self.api_url}/on").json()
            return response
        except Exception as e:
            self.log.error(f"Error turning on laser: {e}")
            return LaserState.OFF

    def off(self) -> LaserState:
        """ Turn off laser.

            @return LaserState: actual laser state
        """
        self.log.debug("Turning off laser")
        try:
            response: LaserState = requests.post(f"{self.api_url}/off").json()
            return response
        except Exception as e:
            self.log.error(f"Error turning off laser: {e}")
            return LaserState.OFF

    def get_laser_state(self) -> LaserState:
        """ Get laser state

        @return LaserState: current laser state
        """
        self.log.debug("Getting laser state")
        try:
            response: LaserState = requests.get(f"{self.api_url}/state").json()
            return response
        except Exception as e:
            self.log.error(f"Error getting laser state: {e}")
            return LaserState.UNKNOWN

    def set_laser_state(self, state: LaserState):
        """ Set laser state.

        @param LaserState state: desired laser state enum
        """
        self.log.debug(f"Setting laser state to {state}")
        try:
            requests.post(f"{self.api_url}/state?state={state}")
        except Exception as e:
            self.log.error(f"Error setting laser state: {e}")

    def get_shutter_state(self) -> ShutterState:
        """ Get laser shutter state

        @return ShutterState: actual laser shutter state
        """
        self.log.debug("Getting shutter state")
        try:
            response: ShutterState = requests.get(f"{self.api_url}/shutter/state").json()
            
            return response
        except Exception as e:
            self.log.error(f"Error getting shutter state: {e}")
            return ShutterState.UNKNOWN

    def set_shutter_state(self, state: ShutterState):
        """ Set laser shutter state.

        @param ShutterState state: desired laser shutter state
            NOTE: Actually Qudi uses bool instead of the
                  ShutterState enum.
        """
        self.log.debug(f"Setting shutter state to {state}")
        try:
            requests.post(f"{self.api_url}/shutter/state?state={int(state)}")
        except Exception as e:
            self.log.error(f"Error setting shutter state: {e}")

    def get_temperatures(self) -> Dict[str, float]:
        """ Get all available temperatures.

        @return dict: dict of temperature names and value in degrees Celsius
        """
        self.log.debug("Getting temperatures")
        try:
            response: Dict[str, float] = requests.get(f"{self.api_url}/temperatures").json()
            return response
        except Exception as e:
            self.log.error(f"Error getting temperatures: {e}")
            return {"psu": 0.0, "head": 0.0}

    def get_extra_info(self) -> str:
        """ Multiple lines of dignostic information

            @return str: extra info
        """
        self.log.debug("Getting extra info")
        try:
            response: str = requests.get(f"{self.api_url}/extra_info").json()
            return response
        except Exception as e:
            self.log.error(f"Error getting extra info: {e}")
            return ""


# Check all Functions of the API. """
# if __name__ == "__main__":
#     laser = SimpleLaserAPI()
# 
#     laser.on_activate()
#     laser.on_deactivate()
#     laser.get_power_range()
#     laser.get_power()
#     laser.get_power_setpoint()
#     laser.set_power(0.1)
#     laser.get_current_unit()
#     laser.get_current_range()
#     laser.get_current()
#     laser.get_current_setpoint()
#     laser.set_current(50)
#     laser.allowed_control_modes()
#     laser.get_control_mode()
#     laser.set_control_mode(ControlMode.POWER)
#     laser.on()
#     laser.off()
#     laser.get_laser_state()
#     laser.set_laser_state(LaserState.ON)
#     laser.get_shutter_state()
#     laser.set_shutter_state(ShutterState.OPEN)
#     laser.get_temperatures()
#     laser.get_extra_info()