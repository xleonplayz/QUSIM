# -*- coding: utf-8 -*-

"""
This file contains the Qudi hardware API for pulsing devices.

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

import asyncio
import websockets
import requests
import json

from typing import List, Union, Dict, Tuple, Set
from enum import IntEnum
import logging

from qudi.core.statusvariable import StatusVar
from qudi.core.configoption import ConfigOption
from qudi.interface.pulser_interface import PulserInterface, PulserConstraints


# - Enums from Interface class - #
class ConnectStatus(IntEnum):
    DISCONNECTED = False
    CONNECTED = True

class PulserStatus(IntEnum):
    UNCONFIGURED = 0
    IDLE = 1
    RUNNING = 2
    PAUSED = 3
    ERROR = -1

class PulserAPI(PulserInterface):
    """ API class for  PulseInterface

    Be careful in adjusting the method names in that class, since some of them
    are also connected to the mwsourceinterface (to give the AWG the possibility
    to act like a microwave source).

    Example config for copy-paste:

    pulser_API:
        module.Class: 'pulser_API.PulserAPI'

    """

    activation_config = StatusVar(default=None)
    force_sequence_option = ConfigOption('force_sequence_option', default=False)
    save_samples = ConfigOption('save_samples', default=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._init_logger()
        self.log.debug("Initializing PulserAPI")

        self.api_url = 'http://localhost:8000/pulser'
        requests.post(f"{self.api_url}/init")


    def _init_logger(self):
        """ Initialize logger for this module. 
            Default level is DEBUG.
        """
        if not hasattr(self, 'log'):
            self.log = logging.getLogger("qudi.hardware.api.pulser_API")
            self.log.setLevel(logging.DEBUG)

            ch = logging.StreamHandler()
            formatter = logging.Formatter('[Pulser] %(asctime)s [%(levelname)s]: %(message)s')
            ch.setFormatter(formatter)
            self.log.addHandler(ch)

    def on_activate(self):
        """ Initialisation performed during activation of the module. """
        self.log.debug("Activating PulserAPI")
        try:
            requests.post(f"{self.api_url}/activate")
        except Exception as e:
            self.log.error(f"Error during activation: {e}")

    def on_deactivate(self):
        """ Deinitialisation performed during deactivation of the module.
        """
        self.log.debug("Deactivating PulserAPI")
        try:
            requests.post(f"{self.api_url}/deactivate")
        except Exception as e:
            self.log.error(f"Error during deactivation: {e}")

    def get_constraints(self) -> PulserConstraints:
        """
        Retrieve the hardware constrains from the Pulsing device.

        @return constraints object: object with pulser constraints as attributes.

        Provides all the constraints (e.g. sample_rate, amplitude, total_length_bins,
        channel_config, ...) related to the pulse generator hardware to the caller.

            SEE PulserConstraints CLASS IN pulser_interface.py FOR AVAILABLE CONSTRAINTS!!!

        If you are not sure about the meaning, look in other hardware files to get an impression.
        If still additional constraints are needed, then they have to be added to the
        PulserConstraints class.

        Each scalar parameter is an ScalarConstraints object defined in cor.util.interfaces.
        Essentially it contains min/max values as well as min step size, default value and unit of
        the parameter.

        PulserConstraints.activation_config differs, since it contain the channel
        configuration/activation information of the form:
            {<descriptor_str>: <channel_set>,
             <descriptor_str>: <channel_set>,
             ...}

        If the constraints cannot be set in the pulsing hardware (e.g. because it might have no
        sequence mode) just leave it out so that the default is used (only zeros).
        """
        self.log.debug("Retrieving pulser constraints")
        try:
            response: Dict = requests.get(f"{self.api_url}/constraints").json()
            constrain = self.__dict2constraint(response)  # Convert dict to PulserConstraints object
            return constrain
        except Exception as e:
            self.log.error(f"Error retrieving constraints: {e}")
            return PulserConstraints()

    def pulser_on(self) -> PulserStatus:
        """ Switches the pulsing device on.

        @return int: error code (0:stopped, -1:error, 1:running)
        """
        self.log.debug("Switching pulser on")
        try:
            response: PulserStatus = requests.post(f"{self.api_url}/on").json()
            return response
        except Exception as e:
            self.log.error(f"Error switching pulser on: {e}")
            return PulserStatus.ERROR

    def pulser_off(self) -> PulserStatus:
        """ Switches the pulsing device off.

        @return int: error code (0:stopped, -1:error, 1:running)
        """
        self.log.debug("Switching pulser off")
        try:
            response: PulserStatus = requests.post(f"{self.api_url}/off").json()
            return response
        except Exception as e:
            self.log.error(f"Error switching pulser off: {e}")
            return PulserStatus.Error

    def write_waveform(self, 
                       name: str, 
                       analog_samples: Dict[str, List[float]], 
                       digital_samples: Dict[str, List[bool]],
                       is_first_chunk: bool, 
                       is_last_chunk: bool,
                       total_number_of_samples: int) -> Tuple[int, List[str]]:
        """
        Write a new waveform or append samples to an already existing waveform on the device memory.
        The flags is_first_chunk and is_last_chunk can be used as indicator if a new waveform should
        be created or if the write process to a waveform should be terminated.

        NOTE: All sample arrays in analog_samples and digital_samples must be of equal length!

        @param str name: the name of the waveform to be created/append to
        @param dict analog_samples: keys are the generic analog channel names (i.e. 'a_ch1') and
                                    values are 1D numpy arrays of type float32 containing the
                                    voltage samples.
        @param dict digital_samples: keys are the generic digital channel names (i.e. 'd_ch1') and
                                     values are 1D numpy arrays of type bool containing the marker
                                     states.
        @param bool is_first_chunk: Flag indicating if it is the first chunk to write.
                                    If True this method will create a new empty wavveform.
                                    If False the samples are appended to the existing waveform.
        @param bool is_last_chunk:  Flag indicating if it is the last chunk to write.
                                    Some devices may need to know when to close the appending wfm.
        @param int total_number_of_samples: The number of sample points for the entire waveform
                                            (not only the currently written chunk)

        @return (int, list): Number of samples written (-1 indicates failed process) and list of
                             created waveform names
        """
        self.log.debug("Writing waveform(s).")
        number_of_samples, waveforms = asyncio.run(
            self.write_waveform_async(name,
                                      analog_samples,
                                      digital_samples,
                                      is_first_chunk,
                                      is_last_chunk,
                                      total_number_of_samples)
            )
        return number_of_samples, waveforms

    async def write_waveform_async(self, 
                             name: str, 
                             analog_samples: Dict[str, List[float]], 
                             digital_samples: Dict[str, List[bool]],
                             is_first_chunk: bool, 
                             is_last_chunk: bool,
                             total_number_of_samples: int) -> Tuple[int, List[str]]:
        uri = "ws://localhost:8000/ws/write_waveform"
        
        async with websockets.connect(uri, max_size=1024*1024*1024) as websocket:
            payload = {
                    "name": name,
                    "analog_samples":  {k: v.tolist() for k, v in analog_samples.items()},
                    "digital_samples":  {k: v.tolist() for k, v in digital_samples.items()},
                    "is_first_chunk": bool(is_first_chunk),
                    "is_last_chunk": bool(is_last_chunk),
                    "total_number_of_samples": int(total_number_of_samples),
                    "module_default_data_dir": self.module_default_data_dir
                }
            await websocket.send(json.dumps(payload))
            response = await websocket.recv()
            result = json.loads(response)
            await websocket.close()  # Close the websocket connection
            return result["number_of_samples"], result["waveforms"]

    def write_sequence(self, name: str, sequence_parameter_list: List[str]) -> int:
        """
        Write a new sequence on the device memory.

        @param name: str, the name of the waveform to be created/append to
        @param sequence_parameter_list: list, contains the parameters for each sequence step and
                                        the according waveform names.

        @return: int, number of sequence steps written (-1 indicates failed process)
        """
        self.log.debug("Writing sequence.")
        try:
            response: int = requests.post(f"{self.api_url}/write_sequence", params={
                'name': name,
                'sequence_parameter_list': sequence_parameter_list,
                'module_default_data_dir': self.module_default_data_dir
            }).json()
            return response
        except Exception as e:
            self.log.error(f"Error writing sequence: {e}")
            return -1

    def get_waveform_names(self) -> Set[str]:
        """ Retrieve the names of all uploaded waveforms on the device.

        @return list: List of all uploaded waveform name strings in the device workspace.
        """
        self.log.debug("Retrieving waveform names")
        try:
            response: Set[str] = requests.get(f"{self.api_url}/get_waveform_names").json()
            return response
        except Exception as e:
            self.log.error(f"Error retrieving waveform names: {e}")
            return []

    def get_sequence_names(self):
        """ Retrieve the names of all uploaded sequence on the device.

        @return list: List of all uploaded sequence name strings in the device workspace.
        """
        self.log.debug("Retrieving sequence names")
        try:
            response: List[str] = requests.get(f"{self.api_url}/get_sequence_names").json()
            return response
        except Exception as e:
            self.log.error(f"Error retrieving sequence names: {e}")
            return []

    def delete_waveform(self, waveform_name: Union[str, List[str]]) -> List[str]:
        """ Delete the waveform with name "waveform_name" from the device memory.

        @param str waveform_name: The name of the waveform to be deleted
                                  Optionally a list of waveform names can be passed.

        @return list: a list of deleted waveform names.
        """
        self.log.debug("Deleting waveform(s).")
        try:
            response: List[str] = requests.get(f"{self.api_url}/delete_waveform", params={
                    'waveform_name': waveform_name
                }).json()
            return response
        except Exception as e:
            self.log.error(f"Error retrieving sequence names: {e}")
            return []

    def delete_sequence(self, sequence_name: Union[str, List[str]]) -> List[str]:
        """ Delete the sequence with name "sequence_name" from the device memory.

        @param str sequence_name: The name of the sequence to be deleted
                                  Optionally a list of sequence names can be passed.

        @return list: a list of deleted sequence names.
        """
        self.log.debug("Deleting sequence(s).")
        try:
            response: List[str] = requests.get(f"{self.api_url}/delete_sequence", params={
                    'sequence_name': sequence_name
                }).json()
            return response
        except Exception as e:
            self.log.error(f"Error deleting sequence: {e}")
            return []

    def load_waveform(self, load_dict: Union[Dict[int, str], List[str]]) -> Dict:
        """ Loads a waveform to the specified channel of the pulsing device.

        @param dict|list load_dict: a dictionary with keys being one of the available channel
                                    index and values being the name of the already written
                                    waveform to load into the channel.
                                    Examples:   {1: rabi_ch1, 2: rabi_ch2} or
                                                {1: rabi_ch2, 2: rabi_ch1}
                                    If just a list of waveform names if given, the channel
                                    association will be invoked from the channel
                                    suffix '_ch1', '_ch2' etc.
                                        {1: rabi_ch1, 2: rabi_ch2}
                                    or
                                        {1: rabi_ch2, 2: rabi_ch1}
                                    If just a list of waveform names if given,
                                    the channel association will be invoked from
                                    the channel suffix '_ch1', '_ch2' etc. A
                                    possible configuration can be e.g.
                                        ['rabi_ch1', 'rabi_ch2', 'rabi_ch3']
        @return dict: Dictionary containing the actually loaded waveforms per
                      channel.

        For devices that have a workspace (i.e. AWG) this will load the waveform
        from the device workspace into the channel. For a device without mass
        memory, this will make the waveform/pattern that has been previously
        written with self.write_waveform ready to play.

        Please note that the channel index used here is not to be confused with the number suffix
        in the generic channel descriptors (i.e. 'd_ch1', 'a_ch1'). The channel index used here is
        highly hardware specific and corresponds to a collection of digital and analog channels
        being associated to a SINGLE wavfeorm asset.
        """
        self.log.debug("Loading waveform(s).")
        try:
            response: Dict[int, str] = requests.post(f"{self.api_url}/load_waveform", json=load_dict).json()
            return response
        except Exception as e:
            self.log.error(f"Error loading waveform: {e}")
            return dict()

    def load_sequence(self, sequence_name: Union[Dict[int, str], List[str]]) -> Dict:
        """ Loads a sequence to the channels of the device in order to be ready for playback.
        For devices that have a workspace (i.e. AWG) this will load the sequence from the device
        workspace into the channels.
        For a device without mass memory this will make the waveform/pattern that has been
        previously written with self.write_waveform ready to play.

        @param dict|list sequence_name: a dictionary with keys being one of the available channel
                                        index and values being the name of the already written
                                        waveform to load into the channel.
                                        Examples:   {1: rabi_ch1, 2: rabi_ch2} or
                                                    {1: rabi_ch2, 2: rabi_ch1}
                                        If just a list of waveform names if given, the channel
                                        association will be invoked from the channel
                                        suffix '_ch1', '_ch2' etc.
        @return dict: Dictionary containing the actually loaded waveforms per channel.
        """
        self.log.debug("Loading sequence(s).")
        try:
            response: Dict = requests.post(f"{self.api_url}/load_sequence", params={
                    'sequence_name': sequence_name
                }).json()
            return response
        except Exception as e:
            self.log.error(f"Error loading sequence: {e}")
            return dict()

    def get_loaded_assets(self) -> Tuple[Dict, Union[str, None]]:
        """
        Retrieve the currently loaded asset names for each active channel of the device.
        The returned dictionary will have the channel numbers as keys.
        In case of loaded waveforms the dictionary values will be the waveform names.
        In case of a loaded sequence the values will be the sequence name appended by a suffix
        representing the track loaded to the respective channel (i.e. '<sequence_name>_1').

        @return (dict, str): Dictionary with keys being the channel number and values being the
                             respective ass
                self.log.error('Unable to determine loaded asset type. Mixed naming convention '
                               'assets loaded (waveform and sequence tracks).')
                return dict(), ''

        return self.current_loaded_assets, asset_typeet loaded into the channel,
                             string describing the asset type ('waveform' or 'sequence')
        """
        self.log.debug("Retrieving loaded assets")
        try:
            response = requests.get(f"{self.api_url}/get_loaded_assets").json()
            loaded_assets: Dict = response[0]
            asset_type: str = response[1]
            return loaded_assets, asset_type
        except Exception as e:
            self.log.error(f"Error retrieving loaded assets: {e}")
            return dict(), ''

    def clear_all(self) -> int:
        """ Clears all loaded waveform from the pulse generators RAM.

        @return int: error code (0:OK, -1:error)

        Unused for digital pulse generators without storage capability
        (PulseBlaster, FPGA).
        """
        self.log.debug("Clearing all loaded assets")
        try:
            response: int = requests.post(f"{self.api_url}/clear_all").json()
            return response
        except Exception as e:
            self.log.error(f"Error clearing all loaded assets: {e}")
            return -1

    def get_status(self) -> Tuple[PulserStatus, Dict[int, str]]:
        """ Retrieves the status of the pulsing hardware

        @return (int, dict): inter value of the current status with the
                             corresponding dictionary containing status
                             description for all the possible status variables
                             of the pulse generator hardware
        """
        self.log.debug("Retrieving pulser status")
        try:
            response = requests.get(f"{self.api_url}/get_status").json()
            current_status: PulserStatus = response[0]
            status_description: Dict[int, str] = response[1]
            return current_status, status_description
        except Exception as e:
            self.log.error(f"Error retrieving pulser status: {e}")
            return PulserStatus.ERROR, {}

    def get_sample_rate(self) -> float:
        """ Get the sample rate of the pulse generator hardware

        @return float: The current sample rate of the device (in Hz)

        Do not return a saved sample rate in a class variable, but instead
        retrieve the current sample rate directly from the device.
        """
        self.log.debug("Retrieving sample rate")
        try:
            response: float = requests.get(f"{self.api_url}/get_sample_rate").json()
            return response
        except Exception as e:
            self.log.error(f"Error retrieving sample rate: {e}")
            return 0.0

    def set_sample_rate(self, sample_rate: float) -> float:
        """ Set the sample rate of the pulse generator hardware

        @param float sample_rate: The sampling rate to be set (in Hz)

        @return float: the sample rate returned from the device.

        Note: After setting the sampling rate of the device, retrieve it again
              for obtaining the actual set value and use that information for
              further processing.
        """
        self.log.debug("Setting sample rate")
        try:
            response: float = requests.post(f"{self.api_url}/set_sample_rate", params={
                'sample_rate': sample_rate
            }).json()
            return response
        except Exception as e:
            self.log.error(f"Error setting sample rate: {e}")
            return 0.0

    def get_analog_level(self, amplitude: List[float]=None, offset: List[float]=None) -> Tuple[Dict[str, float], Dict[str, float]]:
        """ Retrieve the analog amplitude and offset of the provided channels.

        @param list amplitude: optional, if a specific amplitude value (in Volt
                               peak to peak, i.e. the full amplitude) of a
                               channel is desired.
        @param list offset: optional, if a specific high value (in Volt) of a
                            channel is desired.

        @return dict: with keys being the generic string channel names and items
                      being the values for those channels. Amplitude is always
                      denoted in Volt-peak-to-peak and Offset in (absolute)
                      Voltage.

        Note: Do not return a saved amplitude and/or offset value but instead
              retrieve the current amplitude and/or offset directly from the
              device.

        If no entries provided then the levels of all channels where simply
        returned. If no analog channels provided, return just an empty dict.
        Example of a possible input:
            amplitude = ['a_ch1','a_ch4'], offset =[1,3]
        to obtain the amplitude of channel 1 and 4 and the offset
            {'a_ch1': -0.5, 'a_ch4': 2.0} {'a_ch1': 0.0, 'a_ch3':-0.75}
        since no high request was performed.

        The major difference to digital signals is that analog signals are
        always oscillating or changing signals, otherwise you can use just
        digital output. In contrast to digital output levels, analog output
        levels are defined by an amplitude (here total signal span, denoted in
        Voltage peak to peak) and an offset (a value around which the signal
        oscillates, denoted by an (absolute) voltage).

        In general there is no bijective correspondence between
        (amplitude, offset) and (value high, value low)!
        """
        self.log.debug("Retrieving analog level")
        try:
            response = requests.get(f"{self.api_url}/get_analog_level", params={
                'amplitude': amplitude,
                'offset': offset
            }).json()
            amplitude_dict: Dict[str, float] = response[0]
            offset_dict: Dict[str, float] = response[1]
            return amplitude_dict, offset_dict
        except Exception as e:
            self.log.error(f"Error retrieving analog level: {e}")
            return dict(), dict()

    def set_analog_level(self, amplitude: Dict[str, float]=None, offset: Dict[str, float]=None) -> Tuple[Dict[str, float], Dict[str, float]]:
        """ Set amplitude and/or offset value of the provided analog channel.

        @param dict amplitude: dictionary, with key being the channel and items
                               being the amplitude values (in Volt peak to peak,
                               i.e. the full amplitude) for the desired channel.
        @param dict offset: dictionary, with key being the channel and items
                            being the offset values (in absolute volt) for the
                            desired channel.

        @return (dict, dict): tuple of two dicts with the actual set values for
                              amplitude and offset.

        If nothing is passed then the command will return two empty dicts.

        Note: After setting the analog and/or offset of the device, retrieve
              them again for obtaining the actual set value(s) and use that
              information for further processing.

        The major difference to digital signals is that analog signals are
        always oscillating or changing signals, otherwise you can use just
        digital output. In contrast to digital output levels, analog output
        levels are defined by an amplitude (here total signal span, denoted in
        Voltage peak to peak) and an offset (a value around which the signal
        oscillates, denoted by an (absolute) voltage).

        In general there is no bijective correspondence between
        (amplitude, offset) and (value high, value low)!
        """
        self.log.debug("Setting analog level")
        try:
            response = requests.post(f"{self.api_url}/set_analog_level", params={
                'amplitude': amplitude,
                'offset': offset
            }).json()
            amplitude_dict: Dict[str, float] = response[0]
            offset_dict: Dict[str, float] = response[1]
            return amplitude_dict, offset_dict
        except Exception as e:
            self.log.error(f"Error setting analog level: {e}")
            return dict(), dict()

    def get_digital_level(self, low: List[float]=None, high: List[float]=None)-> Tuple[Dict[str, float], Dict[str, float]]:
        """ Retrieve the digital low and high level of the provided channels.

        @param list low: optional, if a specific low value (in Volt) of a
                         channel is desired.
        @param list high: optional, if a specific high value (in Volt) of a
                          channel is desired.

        @return: (dict, dict): tuple of two dicts, with keys being the channel
                               number and items being the values for those
                               channels. Both low and high value of a channel is
                               denoted in (absolute) Voltage.

        Note: Do not return a saved low and/or high value but instead retrieve
              the current low and/or high value directly from the device.

        If no entries provided then the levels of all channels where simply
        returned. If no digital channels provided, return just an empty dict.

        Example of a possible input:
            low = ['d_ch1', 'd_ch4']
        to obtain the low voltage values of digital channel 1 an 4. A possible
        answer might be
            {'d_ch1': -0.5, 'd_ch4': 2.0} {}
        since no high request was performed.

        The major difference to analog signals is that digital signals are
        either ON or OFF, whereas analog channels have a varying amplitude
        range. In contrast to analog output levels, digital output levels are
        defined by a voltage, which corresponds to the ON status and a voltage
        which corresponds to the OFF status (both denoted in (absolute) voltage)

        In general there is no bijective correspondence between
        (amplitude, offset) and (value high, value low)!
        """
        self.log.debug("Retrieving digital level")
        try:
            response = requests.get(f"{self.api_url}/get_digital_level", params={
                'low': low,
                'high': high
            }).json()
            low_dict: Dict[str, float] = response[0]
            high_dict: Dict[str, float] = response[1]
            return low_dict, high_dict
        except Exception as e:
            self.log.error(f"Error retrieving digital level: {e}")
            return dict(), dict()

    def set_digital_level(self, low: Dict[str, float]=None, high: Dict[str, float]=None) -> Tuple[Dict[str, float], Dict[str, float]]:
        """ Set low and/or high value of the provided digital channel.

        @param dict low: dictionary, with key being the channel and items being
                         the low values (in volt) for the desired channel.
        @param dict high: dictionary, with key being the channel and items being
                         the high values (in volt) for the desired channel.

        @return (dict, dict): tuple of two dicts where first dict denotes the
                              current low value and the second dict the high
                              value.

        If nothing is passed then the command will return two empty dicts.

        Note: After setting the high and/or low values of the device, retrieve
              them again for obtaining the actual set value(s) and use that
              information for further processing.

        The major difference to analog signals is that digital signals are
        either ON or OFF, whereas analog channels have a varying amplitude
        range. In contrast to analog output levels, digital output levels are
        defined by a voltage, which corresponds to the ON status and a voltage
        which corresponds to the OFF status (both denoted in (absolute) voltage)

        In general there is no bijective correspondence between
        (amplitude, offset) and (value high, value low)!
        """
        self.log.debug("Setting digital level")
        try:
            response = requests.post(f"{self.api_url}/set_digital_level", params={
                'low': low,
                'high': high
            }).json()
            low_dict: Dict[str, float] = response[0]
            high_dict: Dict[str, float] = response[1]
            return low_dict, high_dict
        except Exception as e:
            self.log.error(f"Error setting digital level: {e}")
            return dict(), dict()

    def get_active_channels(self, ch: List[str]=None) -> Dict[str, bool]:
        """ Get the active channels of the pulse generator hardware.

        @param list ch: optional, if specific analog or digital channels are
                        needed to be asked without obtaining all the channels.

        @return dict:  where keys denoting the channel number and items boolean
                       expressions whether channel are active or not.

        Example for an possible input (order is not important):
            ch = ['a_ch2', 'd_ch2', 'a_ch1', 'd_ch5', 'd_ch1']
        then the output might look like
            {'a_ch2': True, 'd_ch2': False, 'a_ch1': False, 'd_ch5': True, 'd_ch1': False}

        If no parameters are passed to this method all channels will be asked
        for their setting.
        """
        self.log.debug("Retrieving active channels")
        try: 
            response: Dict[str, bool] = requests.get(f"{self.api_url}/get_active_channels", params={
                'ch': ch
            }).json()
            return response
        except Exception as e:
            self.log.error(f"Error retrieving active channels: {e}")
            return dict()

    def set_active_channels(self, ch: Dict[str, bool]=None) -> Dict[str, bool]:
        """
        Set the active/inactive channels for the pulse generator hardware.
        The state of ALL available analog and digital channels will be returned
        (True: active, False: inactive).
        The actually set and returned channel activation must be part of the available
        activation_configs in the constraints.
        You can also activate/deactivate subsets of available channels but the resulting
        activation_config must still be valid according to the constraints.
        If the resulting set of active channels can not be found in the available
        activation_configs, the channel states must remain unchanged.

        @param dict ch: dictionary with keys being the analog or digital string generic names for
                        the channels (i.e. 'd_ch1', 'a_ch2') with items being a boolean value.
                        True: Activate channel, False: Deactivate channel

        @return dict: with the actual set values for ALL active analog and digital channels

        If nothing is passed then the command will simply return the unchanged current state.

        Note: After setting the active channels of the device, use the returned dict for further
              processing.

        Example for possible input:
            ch={'a_ch2': True, 'd_ch1': False, 'd_ch3': True, 'd_ch4': True}
        to activate analog channel 2 digital channel 3 and 4 and to deactivate
        digital channel 1. All other available channels will remain unchanged.
        """
        self.log.debug("Setting active channels")
        try:
            response: Dict[str, bool] = requests.post(f"{self.api_url}/set_active_channels", params={
                'ch': ch
            }).json()
            return response
        except Exception as e:
            self.log.error(f"Error setting active channels: {e}")
            return dict()

    def get_interleave(self) -> bool:
        """ Check whether Interleave is ON or OFF in AWG.

        @return bool: True: ON, False: OFF

        Unused for pulse generator hardware other than an AWG.
        """
        self.log.debug('Get interleave status.')
        try:
            response: bool = requests.get(f"{self.api_url}/get_interleave").json()
            return response
        except Exception as e:
            self.log.error(f'Error while getting interleave status: {e}')
            return False

    def set_interleave(self, state: bool=False) -> bool:
        """ Turns the interleave of an AWG on or off.

        @param bool state: The state the interleave should be set to
                           (True: ON, False: OFF)

        @return bool: actual interleave status (True: ON, False: OFF)

        Note: After setting the interleave of the device, retrieve the
              interleave again and use that information for further processing.

        Unused for pulse generator hardware other than an AWG.
        """
        self.log.debug('Set interleave status.')
        try:
            response: bool = requests.post(f"{self.api_url}/set_interleave", params={
                'state': state
            }).json()
            return response
        except Exception as e:
            self.log.error(f'Error while setting interleave status: {e}')
            return False

    def write(self, command: str) -> int:
        """ Sends a command string to the device.

        @param string command: string containing the command

        @return int: error code (0:OK, -1:error)
        """
        self.log.debug('Sending command: {0}'.format(command))
        try:
            response: int = requests.post(f"{self.api_url}/write", param={
                    'command': command
                }).json()
            return response
        except Exception as e:
            self.log.error(f"Error sending command: {e}")
            return -1

    def query(self, question: str) -> str:
        """ Asks the device a 'question' and receive and return an answer from it.

        @param string question: string containing the command

        @return string: the answer of the device to the 'question' in a string
        """
        self.log.debug('Asking question: {0}'.format(question))
        try:
            response: str = requests.get(f"{self.api_url}/query", params={
                'question': question
            }).json()
            return response
        except Exception as e:
            self.log.error(f"Error querying device: {e}")
            return ""

    def reset(self):
        """ Reset the device.

        @return int: error code (0:OK, -1:error)
        """
        self.log.debug("Resetting device")
        try:
            response: int = requests.post(f"{self.api_url}/reset").json()
            return response
        except Exception as e:
            self.log.error(f"Error resetting device: {e}")
            return -1


    ### --- Helper Functions --- ###
    def __dict2constraint_legacy(self, response) -> PulserConstraints:
        """ Convert a dictionary to a PulserConstraints object.

        @param response: dict, the response from the API containing constraints data

        @return PulserConstraints: an instance of PulserConstraints with the data from the response
        """
        constraint = PulserConstraints()
        constraint.sample_rate = response['sample_rate']

        constraint.a_ch_amplitude = response['a_ch_amplitude']
        constraint.a_ch_offset = response['a_ch_offset']

        constraint.d_ch_low = response['d_ch_low']
        constraint.d_ch_high = response['d_ch_high']

        constraint.waveform_length = response['waveform_length']

        constraint.waveform_num = response['waveform_num']
        constraint.sequence_num = response['sequence_num']
        constraint.subsequence_num = response['subsequence_num']

        constraint.sequence_steps = response['sequence_steps']
        constraint.repetitions = response['repetitions']
        constraint.event_triggers = response['event_triggers']
        constraint.flags = response['flags']

        # activation_config is of type Dict[str, Set[str]]
        # thus convert Dict[str, List[str]] to Dict[str, Set[str]]
        dict_set = {k: set(v) for k, v in response['activation_config'].items()}
        constraint.activation_config = dict_set
        constraint.sequence_option = response['sequence_option']

        return constraint
    
    def __dict2constraint(self, response) -> PulserConstraints:
        pc = PulserConstraints()
        # ScalarConstraint-Fields
        for key in [
                "sample_rate", 
                "a_ch_amplitude", "a_ch_offset", 
                "d_ch_low", "d_ch_high",
                "waveform_length", 
                "waveform_num", "sequence_num", "subsequence_num",
                "sequence_steps", "repetitions"
            ]:
            if key in response:
                pc.__dict__[key].min = response[key]['_minimum']
                pc.__dict__[key].max = response[key]['_maximum']
                pc.__dict__[key].step = response[key]['_increment']
                pc.__dict__[key].default = response[key]['_default']
                # pc.__dict__[key].enforce_int = response[key]['_enforce_int']

        #Lists
        for key in ["event_triggers", "flags"]:
            if key in response:
                setattr(pc, key, response[key])

        # Other
        if "activation_config" in response:
            # activation_config is of type Dict[str, Set[str]]
            # thus convert Dict[str, List[str]] to Dict[str, Set[str]]
            dict_set = {k: set(v) for k, v in response['activation_config'].items()}
            pc.activation_config = dict_set
        if "sequence_option" in response:
            pc.sequence_option = response["sequence_option"]
        return pc