from fastapi import APIRouter, WebSocket

from pydantic import BaseModel, Field
from enum import IntEnum, Enum
from typing import List, FrozenSet, Dict, Union, Set, Tuple

import numpy as np
import datetime
import os

import logging
from qudi.util.datastorage import get_timestamp_filename, create_dir_for_file
from qudi.util.helpers import natural_sort
from qudi.util.yaml import yaml_dump

from qudi.interface.pulser_interface import PulserConstraints, SequenceOption


### --- Models for Pulser Control --- ###
# - Enums from Interface class - #
class ConnectStatus(Enum):
    DISCONNECTED = False
    CONNECTED = True

class PulserStatus(IntEnum):
    UNCONFIGURED = 0
    IDLE = 1
    RUNNING = 2
    PAUSED = 3
    ERROR = -1



# - Pydantic Model for Pulser Data - #
class PulserModel(BaseModel):
    # Parameters
    sample_rate: float = Field(..., description="Sample rate of the pulser in Hz")

    # Status Variables
    connected: ConnectStatus = Field(..., description="Connection status of the pulser")
    current_status: PulserStatus = Field(..., description="Current status of the pulser")
    
    # Channel Configuration
    amplitude_dict: Dict[str, float] = Field(..., description="Max Amplitude for analog channels in volts")
    offset_dict: Dict[str, float] = Field(..., description="Offset for analog channels in volts")
    
    digital_high_dict: Dict[str, float] = Field(..., description="Digital high values for digital channels in volts")
    digital_low_dict: Dict[str, float] = Field(..., description="Digital low values for digital channels in volts")
    
    activation_config: Dict[str, FrozenSet[str]] = Field(..., description="Configuration for active channels")
    
    channel_states: Dict[str, bool] = Field(..., description="States of the channels, True for active, False for inactive")
    current_loaded_assets: Dict[int, str] = Field(..., description="Currently loaded assets for each channel, where key is channel number and value is waveform name")
    
    # Waveforms
    waveform_set: Set[str] = Field(..., description="Set of all available waveform names")
    
    # Sequences
    sequence_dict: Dict[str, str] = Field(..., description="Dictionary of sequences")
    
    # Other
    interleave: bool = Field(..., description="Indicates if interleaving is used in sequences")
    use_sequencer: bool = Field(..., description="Indicates if the sequencer is used for sequences")
    force_sequence_option: bool = Field(...,  description="Force the use of sequence options even if not required")
    save_samples: bool = Field(..., description="Indicates if samples should be saved to disk")


### --- FastAPI Application Setup --- ###
router = APIRouter()

# Initial State of the Pulser
pulser_data = PulserModel(
    sample_rate=0,

    connected=ConnectStatus.DISCONNECTED,
    current_status=PulserStatus.UNCONFIGURED,

    amplitude_dict=dict(),
    offset_dict=dict(),

    digital_high_dict=dict(),
    digital_low_dict=dict(),

    activation_config=dict(),
    channel_states=dict(),
    current_loaded_assets=dict(),

    waveform_set=set(),

    sequence_dict=dict(),
    
    interleave=False,
    use_sequencer=True,
    force_sequence_option=False,
    save_samples=True
)

@router.post("/pulser/init")
def initialize_pulser():
    """ Initialize the pulser with default values. """
    pulser_data.connected = ConnectStatus.DISCONNECTED
    pulser_data.sample_rate = 25e9

    # Deactivate all channels at first:
    pulser_data.channel_states = {'a_ch1': False, 'a_ch2': False, 'a_ch3': False,
                                  'd_ch1': False, 'd_ch2': False, 'd_ch3': False, 'd_ch4': False,
                                  'd_ch5': False, 'd_ch6': False, 'd_ch7': False, 'd_ch8': False}

    # for each analog channel one value
    pulser_data.amplitude_dict = {'a_ch1': 1.0, 'a_ch2': 1.0, 'a_ch3': 1.0}
    pulser_data.offset_dict    = {'a_ch1': 0.0, 'a_ch2': 0.0, 'a_ch3': 0.0}

    # for each digital channel one value
    pulser_data.digital_high_dict = {'d_ch1': 5.0, 'd_ch2': 5.0, 'd_ch3': 5.0, 'd_ch4': 5.0,
                                     'd_ch5': 5.0, 'd_ch6': 5.0, 'd_ch7': 5.0, 'd_ch8': 5.0}
    pulser_data.digital_low_dict  = {'d_ch1': 0.0, 'd_ch2': 0.0, 'd_ch3': 0.0, 'd_ch4': 0.0,
                                     'd_ch5': 0.0, 'd_ch6': 0.0, 'd_ch7': 0.0, 'd_ch8': 0.0}

    pulser_data.waveform_set = set()
    pulser_data.sequence_dict = dict()

    pulser_data.current_loaded_assets = dict()

    pulser_data.use_sequencer = True
    pulser_data.interleave = False

    pulser_data.current_status = PulserStatus.UNCONFIGURED


@router.post("/pulser/activate")
def on_activate():
    """ Connect to the pulser. """
    pulser_data.connected = ConnectStatus.CONNECTED

    constraints = get_constraints()

    if pulser_data.activation_config is None:
        pulser_data.activation_config = constraints.activation_config['config0']
    elif pulser_data.activation_config not in constraints.activation_config.values():
        pulser_data.activation_config = constraints.activation_config['config0']

    for chnl in pulser_data.activation_config:
        pulser_data.channel_states[chnl] = True

@router.post("/pulser/deactivate",)
def on_deactivate():
    """ Disconnect from the pulser. """
    pulser_data.connected = ConnectStatus.DISCONNECTED

@router.get("/pulser/constraints")
def get_constraints():
    """ Get the constraints for the pulser. """
    constraints = __createConstraints()
    return constraints

@router.post("/pulser/on", response_model=PulserStatus)
def pulser_on() -> PulserStatus:
    """ Switches the pulsing device on. """
    if pulser_data.current_status == PulserStatus.UNCONFIGURED:
        pulser_data.current_status = PulserStatus.IDLE
        return pulser_data.current_status
    else:
        raise RuntimeError(f"Pulser is already on with status {pulser_data.current_status}.")

@router.post("/pulser/off", response_model=PulserStatus)
def pulser_off() -> PulserStatus:
    """ Switches the pulsing device off. """
    pulser_data.current_status = PulserStatus.UNCONFIGURED
    return pulser_data.current_status

@router.websocket("/ws/write_waveform")
async def write_waveform(websocket: WebSocket):
    await websocket.accept()
    data = await websocket.receive_json()

    name = data['name']
    analog_samples = data['analog_samples']
    analog_samples = {k: np.array(v) for k, v in analog_samples.items()}
    digital_samples = data['digital_samples']
    digital_samples = {k: np.array(v) for k, v in digital_samples.items()}
    is_first_chunk = data['is_first_chunk']
    is_last_chunk = data['is_last_chunk']
    total_number_of_samples = data['total_number_of_samples']
    module_default_data_dir = data['module_default_data_dir']

    waveforms: List[int] = list()

    # Sanity checks
    if len(analog_samples) > 0:
        number_of_samples = len(analog_samples[list(analog_samples)[0]])
    elif len(digital_samples) > 0:
        number_of_samples = len(digital_samples[list(digital_samples)[0]])
    else:
        logging.error('No analog or digital samples passed to write_waveform method in API pulser.')
        await websocket.send_json({"number_of_samples": -1, "waveforms": waveforms})

    for chnl, samples in analog_samples.items():
        if len(samples) != number_of_samples:
            logging.error('Unequal length of sample arrays for different channels in API pulser.')
            await websocket.send_json({"number_of_samples": -1, "waveforms": waveforms})
    for chnl, samples in digital_samples.items():
        if len(samples) != number_of_samples:
            logging.error('Unequal length of sample arrays for different channels in API pulser.')
            await websocket.send_json({"number_of_samples": -1, "waveforms": waveforms})

    # Determine if only digital samples are active. In that case each channel will get a
    # waveform. Otherwise only the analog channels will have a waveform with digital channel
    # samples included (as it is the case in Tektronix and Keysight AWGs).
    # Simulate a 1Gbit/s transfer speed. Assume each analog waveform sample is 5 bytes large
    # (4 byte float and 1 byte marker bitmask). Assume each digital waveform sample is 1 byte.

    if not pulser_data.save_samples:
        if len(analog_samples) > 0:
            for chnl in analog_samples:
                waveforms.append(name + chnl[1:])
                # time.sleep(number_of_samples * 5 * 8 / 1024 ** 3)
        else:
            for chnl in digital_samples:
                waveforms.append(name + chnl[1:])
                # time.sleep(number_of_samples * 8 / 1024 ** 3)
    else:
        dt = datetime.datetime.now()

        if len(analog_samples) > 0:
            for chnl in analog_samples:
                waveforms.append(name + chnl[1:])
        else:
            for chnl in digital_samples:
                waveforms.append(name + chnl[1:])

        saved = dict()
        for chnl in analog_samples:
            saved[name + '_' + chnl] = analog_samples[chnl]
            logging.debug(f'Adding channel {name} {chnl} with shape {analog_samples[chnl].shape} and type {analog_samples[chnl].dtype} for saving.')

        for chnl in digital_samples:
            saved[name + '_' + chnl] = digital_samples[chnl]
            logging.debug(f'Adding channel {name} {chnl} with shape {digital_samples[chnl].shape} and type {digital_samples[chnl].dtype} for saving.')

        filename = get_timestamp_filename(timestamp=datetime.datetime.now()) + '_waveform.npz'
        file_path = os.path.join(module_default_data_dir, filename)
        create_dir_for_file(file_path)
        np.savez_compressed(file_path, **saved)
        logging.debug(f'Saving {name} took {datetime.datetime.now() - dt}')

    pulser_data.waveform_set.update(waveforms)

    logging.info('Waveforms with nametag "{0}" directly written on API pulser.'.format(name))
    
    await websocket.send_json({"number_of_samples": number_of_samples, "waveforms": waveforms})
    await websocket.close()

@router.post("/pulser/write_sequence", response_model=int)
def write_sequence(name: str, 
                   sequence_parameter_list: List[str],
                   module_default_data_dir: str) -> int:
    # Check if all waveforms are present on virtual device memory
    for waveform_tuple, param_dict in sequence_parameter_list:
        for waveform in waveform_tuple:
            if waveform not in pulser_data.waveform_set:
                logging.error('Failed to create sequence "{0}" due to waveform "{1}" not '
                               'present in device memory.'.format(name, waveform))
                return -1

    if name in pulser_data.sequence_dict:
        del pulser_data.sequence_dict[name]

    pulser_data.sequence_dict[name] = len(sequence_parameter_list[0][0])
    if not pulser_data.save_samples:
        pass
    else:
        filename = get_timestamp_filename(timestamp=datetime.datetime.now()) + '_sequence.txt'
        file_path = os.path.join(module_default_data_dir, filename)
        dump_list = [(tuple(wfm), dict(param)) for wfm, param in sequence_parameter_list]
        yaml_dump(file_path, dump_list)

    logging.info('Sequence with name "{0}" directly written on API pulser.'.format(name))
    return len(sequence_parameter_list)

@router.get("/pulser/get_waveform_names", response_model=Set[str])
def get_waveform_names() -> Set[str]:
    """ Get the names of the available waveforms. """
    return pulser_data.waveform_set

@router.get("/pulser/get_sequence_names", response_model=List[str])
def get_sequence_names() -> List[str]:
    """ Get the names of the available sequences. """
    return list(pulser_data.sequence_dict)

@router.post("/pulser/delete_waveform", response_model=List[str])
def delete_waveform(waveform_name: Union[str, List[str]]) -> List[str]:
    """ Delete the waveform with the given name from the pulser's memory. """
    if isinstance(waveform_name, str):
        waveform_name = [waveform_name]

    deleted_waveforms: List[str] = list()
    for waveform in waveform_name:
        if waveform in pulser_data.waveform_set:
            pulser_data.waveform_set.remove(waveform)
            deleted_waveforms.append(waveform)

    return deleted_waveforms

@router.post("/pulser/delete_sequence", response_model=List[str])
def delete_sequence(sequence_name: Union[str, List[str]]):
    """ Delete the sequence with the given name from the pulser's memory. """
    if isinstance(sequence_name, str):
        sequence_name = [sequence_name]

    deleted_sequences = list()
    for sequence in sequence_name:
        if sequence in pulser_data.sequence_dict:
            pulser_data.sequence_dict.remove(sequence)
            deleted_sequences.append(sequence)

    return deleted_sequences

@router.post("/pulser/load_waveform", response_model=Dict)
def load_waveform(load_dict: Union[Dict[int, str], List[str]]) -> Dict:
    if isinstance(load_dict, list):
        new_dict = dict()
        for waveform in load_dict:
            channel = int(waveform.rsplit('_ch', 1)[1])
            new_dict[channel] = waveform
        load_dict = new_dict

    # Determine if the device is purely digital and get all active channels
    analog_channels = [chnl for chnl in pulser_data.activation_config if chnl.startswith('a')]
    digital_channels = [chnl for chnl in pulser_data.activation_config if chnl.startswith('d')]
    pure_digital = len(analog_channels) == 0

    # Check if waveforms are present in virtual API device memory and specified channels are
    # active. Create new load dict.
    new_loaded_assets = dict()
    for channel, waveform in load_dict.items():
        if waveform not in pulser_data.waveform_set:
            logging.error('Loading failed. Waveform "{0}" not found on device memory.'
                           ''.format(waveform))
            return pulser_data.current_loaded_assets
        if pure_digital:
            if 'd_ch{0:d}'.format(channel) not in digital_channels:
                logging.error('Loading failed. Digital channel {0:d} not active.'
                               ''.format(channel))
                return pulser_data.current_loaded_assets
        else:
            if 'a_ch{0:d}'.format(channel) not in analog_channels:
                logging.error('Loading failed. Analog channel {0:d} not active.'
                               ''.format(channel))
                return pulser_data.current_loaded_assets
        new_loaded_assets[channel] = waveform

    pulser_data.current_loaded_assets = new_loaded_assets

    return get_loaded_assets()[0]

@router.post("/pulser/load_sequence", response_model=dict)
def load_sequence(sequence_name: Union[Dict[int, str], List[str]]) -> Dict:
    """ Load sequences into the pulser. """
    if sequence_name not in pulser_data.sequence_dict:
        logging.error('Sequence loading failed. No sequence with name "{0}" found on device '
                       'memory.'.format(sequence_name))
        return get_loaded_assets()[0]

    # Determine if the device is purely digital and get all active channels
    analog_channels = natural_sort(chnl for chnl in pulser_data.activation_config if chnl.startswith('a'))
    digital_channels = natural_sort(chnl for chnl in pulser_data.activation_config if chnl.startswith('d'))
    pure_digital = len(analog_channels) == 0

    if pure_digital and len(digital_channels) != pulser_data.sequence_dict[sequence_name]:
        logging.error('Sequence loading failed. Number of active digital channels ({0:d}) does'
                       ' not match the number of tracks in the sequence ({1:d}).'
                       ''.format(len(digital_channels), pulser_data.sequence_dict[sequence_name]))
        return get_loaded_assets()[0]
    if not pure_digital and len(analog_channels) != pulser_data.sequence_dict[sequence_name]:
        logging.error('Sequence loading failed. Number of active analog channels ({0:d}) does'
                       ' not match the number of tracks in the sequence ({1:d}).'
                       ''.format(len(analog_channels), pulser_data.sequence_dict[sequence_name]))
        return get_loaded_assets()[0]

    new_loaded_assets = dict()
    if pure_digital:
        for track_index, chnl in enumerate(digital_channels):
            chnl_num = int(chnl.split('ch')[1])
            new_loaded_assets[chnl_num] = sequence_name
    else:
        for track_index, chnl in enumerate(analog_channels):
            chnl_num = int(chnl.split('ch')[1])
            new_loaded_assets[chnl_num] = sequence_name

    pulser_data.current_loaded_assets = new_loaded_assets
    return get_loaded_assets()[0]

@router.get("/pulser/get_loaded_assets", response_model=Tuple[Dict, Union[str, None]])
def get_loaded_assets() -> Tuple[Dict, Union[str, None]]:
    """ Get the currently loaded assets in the pulser. """
    asset_type = None
    for asset_name in pulser_data.current_loaded_assets.values():
        if 'ch' in asset_name.rsplit('_', 1)[1]:
            current_type = 'waveform'
        else:
            current_type = 'sequence'

        if asset_type is None or asset_type == current_type:
            asset_type = current_type
        else:
            logging.error('Unable to determine loaded asset type. Mixed naming convention '
                           'assets loaded (waveform and sequence tracks).')
            return dict(), ''
    return pulser_data.current_loaded_assets, asset_type

@router.post("/pulser/clear_all")
def clear_all() -> int:
    """ Clear all loaded assets in the pulser. """
    pulser_data.current_loaded_assets = dict()
    pulser_data.waveform_set = set()
    pulser_data.sequence_dict = dict()

    logging.info('All loaded assets cleared from the pulser.')
    return 0

@router.get("/pulser/get_status", response_model=Tuple[PulserStatus, Dict[int, str]])
def get_status() -> Tuple[PulserStatus, Dict[int, str]]:
    """ Get the current status of the pulser. """
    status_dic = {-1: 'Failed Request or Communication.', 
                   0: 'Device has stopped, but can receive commands.',
                   1: 'Device is active and running.'}
    return pulser_data.current_status, status_dic

@router.get("/pulser/get_sample_rate", response_model=float)
def get_sample_rate() -> float:
    """ Get the sample rate of the pulser. """
    return pulser_data.sample_rate

@router.post("/pulser/set_sample_rate", response_model=float)
def set_sample_rate(sample_rate: float) -> float:
    """ Set the sample rate of the pulser. """
    constraint = get_constraints().sample_rate
    
    if sample_rate > constraint.max:
        pulser_data.sample_rate = constraint.max
    elif sample_rate < constraint.min:
        pulser_data.sample_rate = constraint.min
    else:
        pulser_data.sample_rate = sample_rate
        
    return pulser_data.sample_rate

@router.get("/pulser/get_analog_level", response_model=Tuple[Dict[str, float], Dict[str, float]])
def get_analog_level(amplitude: List[float]=None, offset: List[float]=None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """ Get the analog amplitude and offset of the pulser channels. """
    if amplitude is None:
        amplitude = []
    if offset is None:
        offset = []

    ampl = dict()
    off = dict()

    if not amplitude and not offset:

        for a_ch, pp_amp in pulser_data.amplitude_dict.items():
            ampl[a_ch] = pp_amp

        for a_ch, offset in pulser_data.offset_dict.items():
            off[a_ch] = offset

    else:
        for a_ch in amplitude:
            ampl[a_ch] = pulser_data.amplitude_dict[a_ch]

        for a_ch in offset:
            off[a_ch] = pulser_data.offset_dict[a_ch]

    return ampl, off

@router.post("/pulser/set_analog_level", response_model=Tuple[Dict[str, float], Dict[str, float]])
def set_analog_level(amplitude: Dict[str, float]=None, offset: Dict[str, float]=None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """ Set the analog amplitude and offset of the pulser channels. """
    if amplitude is None:
        amplitude = dict()
    if offset is None:
        offset = dict()

    for a_ch, amp in amplitude.items():
        pulser_data.amplitude_dict[a_ch] = amp

    for a_ch, off in offset.items():
        pulser_data.offset_dict[a_ch] = off

    ampl, off = get_analog_level(amplitude=list(amplitude), offset=list(offset))
    return ampl, off

@router.get("/pulser/get_digital_level", response_model=Tuple[Dict[str, float], Dict[str, float]])
def get_digital_level(low: List[str]=None, high: List[str]=None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """ Get the digital low and high levels of the pulser channels. """
    if low is None:
        low = []
    if high is None:
        high = []

    if not low and not high:
        low_val = pulser_data.digital_low_dict
        high_val = pulser_data.digital_high_dict
    else:
        low_val = dict()
        high_val = dict()
        for d_ch in low:
            low_val[d_ch] = pulser_data.digital_low_dict[d_ch]
        for d_ch in high:
            high_val[d_ch] = pulser_data.digital_high_dict[d_ch]

    return low_val, high_val

@router.post("/pulser/set_digital_level", response_model=Tuple[Dict[str, float], Dict[str, float]])
def set_digital_level(low: Dict[str, float]=None, high: Dict[str, float]=None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """ Set the digital low and high levels of the pulser channels. """
    if low is None:
        low = dict()
    if high is None:
        high = dict()

    for d_ch, low_val in low.items():
        pulser_data.digital_low_dict[d_ch] = low_val

    for d_ch, high_val in high.items():
        pulser_data.digital_high_dict[d_ch] = high_val

    low_val, high_val = get_digital_level(low=list(low), high=list(high))
    return low_val, high_val

@router.get("/pulser/get_active_channels", response_model=Dict[str, bool])
def get_active_channels(ch: List[str]=None) -> Dict[str, bool]:
    """ Get the active channels of the pulser. """
    if ch is None:
        ch = []

    active_ch = {}

    if not ch:
        active_ch = pulser_data.channel_states

    else:
        for channel in ch:
            active_ch[channel] = pulser_data.channel_states[channel]

    return active_ch

@router.post("/pulser/set_active_channels", response_model=Dict[str, bool])
def set_active_channels(ch: Dict[str, bool]=None) -> Dict[str, bool]:
    if ch is None:
        ch = {}
    old_activation = pulser_data.channel_states.copy()
    for channel in ch:
        pulser_data.channel_states[channel] = ch[channel]

    active_channel_set = {chnl for chnl, is_active in pulser_data.channel_states.items() if is_active}
    if active_channel_set not in get_constraints().activation_config.values():
        logging.error('Channel activation to be set not found in constraints.\n'
                       'Channel activation unchanged.')
        pulser_data.channel_states = old_activation
    else:
        pulser_data.activation_config = active_channel_set

    return get_active_channels(ch=list(ch))

@router.get("/pulser/get_interleave", response_model=bool)
def get_interleave() -> bool:
    """ Get the interleave mode status of the pulser. """
    return pulser_data.interleave

@router.post("/pulser/set_interleave", response_model=bool)
def set_interleave(state: bool=False) -> bool:
    """ Set the interleave mode of the pulser. """
    pulser_data.interleave = state
    return pulser_data.interleave

@router.post("/pulser/write", response_model=int)
def write(command: str) -> int:
    logging.info('It is so nice that you talk to me and told me "{0}"; '
                    'as a dummy API it is very dull out here! :) '.format(command))
    return 0

@router.post("/pulser/query", response_model=str)
def query(question: str) -> str:
    """ Query the pulser with a command. """
    logging.info('Dude, I\'m a API! Your question \'{0}\' is way too '
                      'complicated for me :D !'.format(question))
    return 'I am Groooooooooooot!'

@router.post("/pulser/reset", response_model=int)
def reset() -> int:
    """ Reset the pulser to its initial state. """
    initialize_pulser()
    pulser_data.connected = ConnectStatus.CONNECTED
    logging.info('API reset!')
    return 0


### --- Helper Function to Simulate Data Generation --- ###
def __createConstraints() -> PulserConstraints:
    """ Create a PulserConstraints object with default values. """
    constraints = PulserConstraints()
    if get_interleave():
        constraints.sample_rate.min = 12.0e9
        constraints.sample_rate.max = 24.0e9
        constraints.sample_rate.step = 4.0e8
        constraints.sample_rate.default = 24.0e9
    else:
        constraints.sample_rate.min = 10.0e6
        constraints.sample_rate.max = 12.0e9
        constraints.sample_rate.step = 10.0e6
        constraints.sample_rate.default = 12.0e9

    constraints.a_ch_amplitude.min = 0.02
    constraints.a_ch_amplitude.max = 2.0
    constraints.a_ch_amplitude.step = 0.001
    constraints.a_ch_amplitude.default = 2.0

    constraints.a_ch_offset.min = -1.0
    constraints.a_ch_offset.max = 1.0
    constraints.a_ch_offset.step = 0.001
    constraints.a_ch_offset.default = 0.0

    constraints.d_ch_low.min = -1.0
    constraints.d_ch_low.max = 4.0
    constraints.d_ch_low.step = 0.01
    constraints.d_ch_low.default = 0.0

    constraints.d_ch_high.min = 0.0
    constraints.d_ch_high.max = 5.0
    constraints.d_ch_high.step = 0.01
    constraints.d_ch_high.default = 5.0

    constraints.waveform_length.min = 80
    constraints.waveform_length.max = 6_000_000_000
    constraints.waveform_length.step = 1
    constraints.waveform_length.default = 80

    constraints.waveform_num.min = 1
    constraints.waveform_num.max = 32000
    constraints.waveform_num.step = 1
    constraints.waveform_num.default = 1

    constraints.sequence_num.min = 1
    constraints.sequence_num.max = 8000
    constraints.sequence_num.step = 1
    constraints.sequence_num.default = 1

    constraints.subsequence_num.min = 1
    constraints.subsequence_num.max = 4000
    constraints.subsequence_num.step = 1
    constraints.subsequence_num.default = 1

    # If sequencer mode is available then these should be specified
    constraints.repetitions.min = 0
    constraints.repetitions.max = 65539
    constraints.repetitions.step = 1
    constraints.repetitions.default = 0

    constraints.event_triggers = ['A', 'B']
    constraints.flags = ['A', 'B', 'C', 'D']

    constraints.sequence_steps.min = 0
    constraints.sequence_steps.max = 8000
    constraints.sequence_steps.step = 1
    constraints.sequence_steps.default = 0
    
    # the name a_ch<num> and d_ch<num> are generic names, which describe UNAMBIGUOUSLY the
    # channels. Here all possible channel configurations are stated, where only the generic
    # names should be used. The names for the different configurations can be customary chosen.
    activation_config: Dict[str, FrozenSet[str]] = dict()
    activation_config['config0'] = frozenset(
        {'a_ch1', 'd_ch1', 'd_ch2', 'a_ch2', 'd_ch3', 'd_ch4'})
    activation_config['config1'] = frozenset(
        {'a_ch2', 'd_ch1', 'd_ch2', 'a_ch3', 'd_ch3', 'd_ch4'})
    # Usage of channel 1 only:
    activation_config['config2'] = frozenset({'a_ch2', 'd_ch1', 'd_ch2'})
    # Usage of channel 2 only:
    activation_config['config3'] = frozenset({'a_ch3', 'd_ch3', 'd_ch4'})
    # Usage of Interleave mode:
    activation_config['config4'] = frozenset({'a_ch1', 'd_ch1', 'd_ch2'})
    # Usage of only digital channels:
    activation_config['config5'] = frozenset(
        {'d_ch1', 'd_ch2', 'd_ch3', 'd_ch4', 'd_ch5', 'd_ch6', 'd_ch7', 'd_ch8'})
    # Usage of only one analog channel:
    activation_config['config6'] = frozenset({'a_ch1'})
    activation_config['config7'] = frozenset({'a_ch2'})
    activation_config['config8'] = frozenset({'a_ch3'})
    # Usage of only the analog channels:
    activation_config['config9'] = frozenset({'a_ch2', 'a_ch3'})
    constraints.activation_config = activation_config

    constraints.sequence_option = SequenceOption.FORCED if pulser_data.force_sequence_option else SequenceOption.OPTIONAL

    return constraints

