# -*- coding: utf-8 -*-

"""
This file contains the Qudi hardware API for fast counting devices.

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
import numpy as np

from typing import List, Tuple
from enum import IntEnum
import logging

from qudi.core.configoption import ConfigOption
from qudi.interface.fast_counter_interface import FastCounterInterface

# - Enums from Interface class - #
class Status(IntEnum):
    UNCONFIGURED = 0
    IDLE = 1
    RUNNING = 2
    PAUSED = 3
    ERROR = -1

class FastCounterAPI(FastCounterInterface):
    """ Fast Counter API for a API usage.

    Example config for copy-paste:

    fastcounter_API:
        module.Class: 'fast_counter_API.FastCounterAPI'
        options:
            gated: False
            #load_trace: None # path to the saved API trace

    """

    # config option
    _gated = ConfigOption('gated', False, missing='warn')

    def __init__(self, *args, **kwargs):
        """ """
        super().__init__(*args, **kwargs)
        
        self._init_logger()
        self.log.debug("Initializing FastCounterAPI")

        self.api_url = 'http://localhost:8000/fastcounter'
        requests.post(f"{self.api_url}/init", params={'gated': self._gated})


    def _init_logger(self):
        """ Initialize logger for this module. 
            Default level is DEBUG.
        """
        if not hasattr(self, 'log'):
            self.log = logging.getLogger("qudi.hardware.api.fast_counter_API")
            self.log.setLevel(logging.DEBUG)

            ch = logging.StreamHandler()
            formatter = logging.Formatter('[FastCounter] %(asctime)s [%(levelname)s]: %(message)s')
            ch.setFormatter(formatter)
            self.log.addHandler(ch)

    def on_activate(self):
        """ Activate module. """
        self.log.debug("Activating FastCounterAPI")
        try:
            requests.get(f"{self.api_url}/on_activate")
        except Exception as e:
            self.log.error(f"Error during activation: {e}")

    def on_deactivate(self):
        """ Deinitialisation performed during deactivation of the module.
        """
        self.log.debug("Deactivating FastCounterAPI")
        try:
            requests.get(f"{self.api_url}/on_deactivate")
        except Exception as e:
            self.log.error(f"Error during deactivation: {e}")

    def get_constraints(self) -> dict:
        """ Retrieve the hardware constrains from the Fast counting device.

        @return dict: dict with keys being the constraint names as string and
                      items are the definition for the constaints.

         The keys of the returned dictionary are the str name for the constraints
        (which are set in this method).

                    NO OTHER KEYS SHOULD BE INVENTED!

        If you are not sure about the meaning, look in other hardware files to
        get an impression. If still additional constraints are needed, then they
        have to be added to all files containing this interface.

        The items of the keys are again dictionaries which have the generic
        dictionary form:
            {'min': <value>,
             'max': <value>,
             'step': <value>,
             'unit': '<value>'}

        Only the key 'hardware_binwidth_list' differs, since they
        contain the list of possible binwidths.

        If the constraints cannot be set in the fast counting hardware then
        write just zero to each key of the generic dicts.
        Note that there is a difference between float input (0.0) and
        integer input (0), because some logic modules might rely on that
        distinction.

        ALL THE PRESENT KEYS OF THE CONSTRAINTS DICT MUST BE ASSIGNED!
        """

        self.log.debug("Retrieving constraints from FastCounterAPI")
        try:
            response: dict = requests.get(f"{self.api_url}/constraints").json()
            return response
        except Exception as e:
            self.log.error(f"Error retrieving constraints: {e}")
            return {}

    def configure(self, bin_width_s, record_length_s, number_of_gates = 1) -> Tuple[float, float, int]:
        """ Configuration of the fast counter.

        @param float bin_width_s: Length of a single time bin in the time trace
                                  histogram in seconds.
        @param float record_length_s: Total length of the timetrace/each single
                                      gate in seconds.
        @param int number_of_gates: optional, number of gates in the pulse
                                    sequence. Ignore for not gated counter.

        @return tuple(binwidth_s, gate_length_s, number_of_gates):
                    float binwidth_s:    the actual set binwidth in seconds
                    float gate_length_s: the actual set gate length in seconds
                    int number_of_gates: the number of gated, which are accepted
        """
        self.log.debug(f"Configuring FastCounterAPI with bin_width_s={bin_width_s}, record_length_s={record_length_s}, number_of_gates={number_of_gates}")
        try:
            response: Tuple[float, float, int] = requests.post(
                    f"{self.api_url}/configure",
                    params={
                        "bin_width_s": bin_width_s,
                        "record_length_s": record_length_s,
                        "number_of_gates": number_of_gates
                    }
                ).json()
            return response
        except Exception as e:
            self.log.error(f"Error during configuration: {e}")
            return (0.0, 0.0, 0)

    def get_status(self) -> int:
        """ Receives the current status of the Fast Counter and outputs it as
            return value.

        0 = unconfigured
        1 = idle
        2 = running
        3 = paused
        -1 = error state
        """
        self.log.info("Getting status from FastCounterAPI")
        try:
            response: int = requests.get(f"{self.api_url}/get_status").json()
            return response
        except Exception as e:
            self.log.error(f"Error retrieving status: {e}")
            return int(Status.ERROR)

    def start_measure(self):
        """ Start the fast counter. """
        self.log.debug("Starting measurement on FastCounterAPI")
        try:
            requests.post(f"{self.api_url}/measure/start")
        except Exception as e:
            self.log.error(f"Error starting measurement: {e}")

    def pause_measure(self):
        """ Pauses the current measurement.

        Fast counter must be initially in the run state to make it pause.
        """
        self.log.debug("Pausing measurement on FastCounterAPI")
        try:
            requests.post(f"{self.api_url}/measure/pause")
        except Exception as e:
            self.log.error(f"Error pausing measurement: {e}")

    def stop_measure(self):
        """ Stop the fast counter. """
        self.log.debug("Stopping measurement on FastCounterAPI")
        try:
            requests.post(f"{self.api_url}/measure/stop")
        except Exception as e:
            self.log.error(f"Error stopping measurement: {e}")

    def continue_measure(self):
        """ Continues the current measurement.

        If fast counter is in pause state, then fast counter will be continued.
        """
        self.log.debug("Continuing measurement on FastCounterAPI")
        try:
            requests.post(f"{self.api_url}/measure/continue")
        except Exception as e:
            self.log.error(f"Error continuing measurement: {e}")

    def is_gated(self) -> bool:
        """ Check the gated counting possibility.

        @return bool: Boolean value indicates if the fast counter is a gated
                      counter (TRUE) or not (FALSE).
        """

        self.log.debug("Checking if FastCounterAPI is gated")
        try:
            response: bool = requests.get(f"{self.api_url}/is_gated").json()
            return response
        except Exception as e:
            self.log.error(f"Error checking gated status: {e}")
            return False

    def get_binwidth(self) -> float:
        """ Returns the width of a single timebin in the timetrace in seconds.

        @return float: current length of a single bin in seconds (seconds/bin)
        """
        self.log.debug("Getting binwidth from FastCounterAPI")
        try:
            binwidth_s: float = requests.get(f"{self.api_url}/get_binwidth_s").json()
            return binwidth_s
        except Exception as e:
            self.log.error(f"Error retrieving binwidth: {e}")
            return 0

    def get_data_trace(self) -> Tuple[np.ndarray, dict]:
        """ Polls the current timetrace data from the fast counter.

        Return value is a numpy array (dtype = int64).
        The binning, specified by calling configure() in forehand, must be
        taken care of in this hardware class. A possible overflow of the
        histogram bins must be caught here and taken care of.
        If the counter is NOT GATED it will return a tuple (1D-numpy-array, info_dict) with
            returnarray[timebin_index]
        If the counter is GATED it will return a tuple (2D-numpy-array, info_dict) with
            returnarray[gate_index, timebin_index]

        info_dict is a dictionary with keys :
            - 'elapsed_sweeps' : the elapsed number of sweeps
            - 'elapsed_time' : the elapsed time in seconds

        If the hardware does not support these features, the values should be None
        """

        # should generate __pulsedmeasurementlogic.measurement_settings['number_of_lasers'] pulses

        self.log.debug("Getting data trace from FastCounterAPI")
        try:
            response: Tuple[List[int], dict] = requests.get(f"{self.api_url}/data_trace").json()
            data_trace, info_dict = response
            return np.array(data_trace, dtype='int64'), info_dict
        except Exception as e:
            self.log.error(f"Error retrieving data trace: {e}")
            return np.array([]), {'elapsed_sweeps': None, 'elapsed_time': None}
        

    def get_frequency(self) -> float:
        """ Returns the frequency of the fast counter in MHz. """
        self.log.debug("Getting frequency from FastCounterAPI")
        try:
            response: float = requests.get(f"{self.api_url}/frequency").json()
            return response
        except Exception as e:
            self.log.error(f"Error retrieving frequency: {e}")
            return 0.0



# if __name__ == "__main__":
#     # Example usage of the FastCounterAPI
#     fast_counter = FastCounterAPI()
    
    