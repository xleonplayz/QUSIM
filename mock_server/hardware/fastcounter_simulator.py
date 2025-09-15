from fastapi import APIRouter

from pydantic import BaseModel, Field
from enum import IntEnum
from typing import List, Tuple

import numpy as np

# Import NV Simulator Backend
try:
    from .nv_simulator_integration import nv_backend
    NV_BACKEND_AVAILABLE = True
    print("NV Simulator backend loaded successfully")
except ImportError as e:
    print(f"Warning: Could not load NV Simulator backend: {e}")
    NV_BACKEND_AVAILABLE = False


### --- Models for Fast Counter Control --- ###
# - Enums from Interface class - #
class Status(IntEnum):
    UNCONFIGURED = 0
    IDLE = 1
    RUNNING = 2
    PAUSED = 3
    ERROR = -1


# - Pydantic Model for Fast COunter Data - #
class FastCounterModel(BaseModel):
    status: Status =         Field(..., description="Current status of the fast counter")
    binwidth: int =          Field(..., description="Number of clock cycles per bin")
    gateLength_bins: int =   Field(..., description="Length of one individual gate in bins")
    clock_frequency: float = Field(..., description="Clock frequency in Hz")
    gated: bool =            Field(..., description="Indicates if the fast counter is a gated counter (True) or not (False)")
    number_of_gates: int =   Field(..., description="Number of gates in the pulse sequence")

    # NV Simulator specific parameters
    mw_frequency: float =    Field(default=2.87e9, description="Microwave frequency in Hz")
    mw_power: float =        Field(default=8e6, description="Microwave power in Hz (Rabi frequency)")
    mw_duration: float =     Field(default=60e-9, description="Microwave pulse duration in seconds")
    mw_amplitude: float =    Field(default=1.0, description="Microwave amplitude (0-1)")
    

### --- FastAPI Application Setup --- ###
router = APIRouter()

# Initial State of the Fast Counter
fastcounter_data = FastCounterModel(
    status=Status.UNCONFIGURED,
    binwidth=1,                         # in clock cycles
    gateLength_bins=1024,               # in number of bins
    clock_frequency=950e6,              # in Hz
    gated=False,
    number_of_gates=1,
    # NV specific defaults
    mw_frequency=2.87e9,               # 2.87 GHz
    mw_power=8e6,                      # 8 MHz Rabi frequency
    mw_duration=60e-9,                 # 60 ns pulse
    mw_amplitude=1.0                   # full amplitude
)


@router.post("/fastcounter/init")
def initialize_fastcounter(gated: bool = False):
    """ Initialize the fastcounter with default values. """
    # Here you would typically perform initialization logic
    fastcounter_data.gated = gated

@router.get("/fastcounter/on_activate")
def on_activate():
    """ Activate the fastcounter and return its current state. """
    # Here you would typically perform activation logic
    pass

@router.get("/fastcounter/on_deactivate")
def on_deactivate():
    """ Deactivate the fastcounter and return its current state. """
    # Here you would typically perform deactivation logic
    pass

@router.get("/fastcounter/constraints", response_model=dict)
def get_constraints() -> dict:
    """ Return the constraints of the fastcounter. """
    # Example for configuration with default values:
    constraints = dict()

    # List of supported hardware bin widths (time resolution per in seconds bin)
    # Used to specify which bin widths the hardware can be configured for.
    # 'get_binwidth' returns the current binwidth in seconds.
    constraints['hardware_binwidth_list'] = list(2 ** np.linspace(0,5,6) / fastcounter_data.clock_frequency)

    # possible additional constraints can be added here: e.g.
    #   constraints['max_sweep_len'] = 6.8
    #   constraints['max_bins']      = 6.8 * fastcounter_data.clock_frequency

    return constraints

@router.post("/fastcounter/configure", response_model=Tuple[float, float, int])
def configure(bin_width_s, record_length_s, number_of_gates = 1) -> Tuple[float, float, int]:
    """ Configuration of the fastcounter. """

    # print(f"Bin width: {bin_width_s}\nRecord Length: {record_length_s}\nNumber of Gates: {number_of_gates}")

    bin_width_s = float(bin_width_s)
    record_length_s = float(record_length_s)
    number_of_gates = int(number_of_gates)

    # Do nothing if fastcounter is running
    if fastcounter_data.status == Status.RUNNING or fastcounter_data.status == Status.PAUSED:
        binwidth_s = fastcounter_data.binwidth / fastcounter_data.clock_frequency
        gate_length_s = fastcounter_data.gateLength_bins * binwidth_s
        return binwidth_s, gate_length_s, fastcounter_data.number_of_gates

    # set class variables
    fastcounter_data.binwidth = int(np.rint(bin_width_s * fastcounter_data.clock_frequency))

    # calculate the actual binwidth depending on the internal clock:
    binwidth_s = fastcounter_data.binwidth / fastcounter_data.clock_frequency

    fastcounter_data.gateLength_bins = int(np.rint(record_length_s / bin_width_s))
    gate_length_s = fastcounter_data.gateLength_bins * binwidth_s

    fastcounter_data.number_of_gates = number_of_gates

    # Configure NV Simulator Backend
    if NV_BACKEND_AVAILABLE:
        try:
            mw_params = {
                'frequency': fastcounter_data.mw_frequency,
                'power': fastcounter_data.mw_power,
                'duration': fastcounter_data.mw_duration,
                'amplitude': fastcounter_data.mw_amplitude
            }
            nv_backend.configure_measurement(
                bin_width_s=binwidth_s,
                record_length_s=gate_length_s,
                number_of_gates=number_of_gates,
                mw_params=mw_params
            )
            print(f"NV Backend configured: bin={binwidth_s*1e9:.1f}ns, record={gate_length_s*1e6:.1f}μs")
        except Exception as e:
            print(f"Warning: Could not configure NV backend: {e}")

    fastcounter_data.status = Status.IDLE

    return binwidth_s, gate_length_s, number_of_gates

@router.get("/fastcounter/get_status", response_model=int)
def get_status() -> int:
    """ Receives the current status of the Fast Counter and outputs it as return value.

    0 = unconfigured
    1 = idle
    2 = running
    3 = paused
    -1 = error state
    """
    return int(fastcounter_data.status)

@router.post("/fastcounter/measure/start")
def start_measure():
    """ Start the fast counter. """
    if fastcounter_data.status == Status.IDLE:
        fastcounter_data.status = Status.RUNNING
        # Simulate starting measurement logic
        # In a real implementation, this would trigger the hardware to start counting
    else:
        raise ValueError("Fast Counter is not in IDLE state, cannot start measurement.")

@router.post("/fastcounter/measure/pause")
def pause_measure():
    """ Pauses the current measurement. """
    if fastcounter_data.status == Status.RUNNING:
        fastcounter_data.status = Status.PAUSED
        # Simulate pausing measurement logic
        # In a real implementation, this would pause the hardware counting
    else:
        raise ValueError("Fast Counter is not RUNNING, cannot pause measurement.")

@router.post("/fastcounter/measure/stop")
def stop_measure():
    """ Stop the fast counter. """
    fastcounter_data.status = Status.IDLE
    # Simulate stopping measurement logic
    # In a real implementation, this would stop the hardware counting

@router.post("/fastcounter/measure/continue")
def continue_measure():
    """ Continues the current measurement. """
    if fastcounter_data.status == Status.PAUSED:
        fastcounter_data.status = Status.RUNNING
        # Simulate continuing measurement logic
        # In a real implementation, this would resume the hardware counting
    else:
        raise ValueError("Fast Counter is not PAUSED, cannot continue measurement.")

@router.get("/fastcounter/is_gated", response_model=bool)
def is_gated() -> bool:
    """ Check the gated counting possibility.

    Boolean return value indicates if the fast counter is a gated counter
    (TRUE) or not (FALSE).
    """
    return fastcounter_data.gated

@router.get("/fastcounter/get_binwidth_s", response_model=float)
def get_binwidth() -> float:
    """ Returns the width of a single timebin in the timetrace in seconds.

    @return float: current length of a single bin in seconds (seconds/bin)
    """
    return fastcounter_data.binwidth / fastcounter_data.clock_frequency

@router.get("/fastcounter/data_trace", response_model=Tuple[List[int], dict])
def get_data_trace() -> Tuple[List[int], dict]:
    """ Polls the current timetrace data from the fast counter.

    @return List[int]: Current time trace data from the fast counter.
    @return dict: Additional information about the data trace: elapsed sweeps and time.
    """
    if fastcounter_data.status in [Status.RUNNING, Status.PAUSED]:
        # Simulate data trace retrieval logic
        # In a real implementation, this would fetch the data from the hardware
        data = __generateData()
        info_dict = {'elapsed_sweeps': None, 'elapsed_time': None}
        return data, info_dict
    else:
        raise ValueError("Fast Counter is not RUNNING or PAUSED, cannot retrieve data trace.")

@router.get("/fastcounter/frequency", response_model=float)
def get_frequency() -> float:
    """ Get the clock frequency of the fast counter in Hz. """
    return fastcounter_data.clock_frequency

# === NV Simulator specific endpoints ===

@router.get("/fastcounter/mw/frequency", response_model=float)
def get_mw_frequency() -> float:
    """ Get the microwave frequency in Hz. """
    return fastcounter_data.mw_frequency

@router.post("/fastcounter/mw/frequency")
def set_mw_frequency(frequency: float):
    """ Set the microwave frequency in Hz. """
    fastcounter_data.mw_frequency = float(frequency)
    if NV_BACKEND_AVAILABLE:
        nv_backend.update_mw_parameters(frequency_hz=frequency)
    print(f"MW frequency set to {frequency/1e9:.3f} GHz")

@router.get("/fastcounter/mw/power", response_model=float)
def get_mw_power() -> float:
    """ Get the microwave power (Rabi frequency) in Hz. """
    return fastcounter_data.mw_power

@router.post("/fastcounter/mw/power")
def set_mw_power(power: float):
    """ Set the microwave power (Rabi frequency) in Hz. """
    fastcounter_data.mw_power = float(power)
    if NV_BACKEND_AVAILABLE:
        nv_backend.update_mw_parameters(power_hz=power)
    print(f"MW power set to {power/1e6:.1f} MHz")

@router.get("/fastcounter/mw/duration", response_model=float)
def get_mw_duration() -> float:
    """ Get the microwave pulse duration in seconds. """
    return fastcounter_data.mw_duration

@router.post("/fastcounter/mw/duration")
def set_mw_duration(duration: float):
    """ Set the microwave pulse duration in seconds. """
    fastcounter_data.mw_duration = float(duration)
    if NV_BACKEND_AVAILABLE:
        nv_backend.update_mw_parameters(duration_s=duration)
    print(f"MW duration set to {duration*1e9:.1f} ns")

@router.get("/fastcounter/mw/amplitude", response_model=float)
def get_mw_amplitude() -> float:
    """ Get the microwave amplitude (0-1). """
    return fastcounter_data.mw_amplitude

@router.post("/fastcounter/mw/amplitude")
def set_mw_amplitude(amplitude: float):
    """ Set the microwave amplitude (0-1). """
    if amplitude < 0 or amplitude > 1:
        raise ValueError("Amplitude must be between 0 and 1")
    fastcounter_data.mw_amplitude = float(amplitude)
    if NV_BACKEND_AVAILABLE:
        nv_backend.update_mw_parameters(amplitude=amplitude)
    print(f"MW amplitude set to {amplitude:.2f}")

@router.get("/fastcounter/nv/info", response_model=dict)
def get_nv_info() -> dict:
    """ Get information about the last NV simulation. """
    if NV_BACKEND_AVAILABLE:
        return nv_backend.get_measurement_info()
    else:
        return {"error": "NV Simulator backend not available"}


### --- Helper Function to Simulate Data Generation --- ###
def __generateData() -> List[int]:
    """ Generate data using NV Simulator or fallback to dummy data. """
    if NV_BACKEND_AVAILABLE:
        try:
            # Use NV Simulator backend to generate realistic data
            data = nv_backend.run_measurement()
            print(f"Generated {len(data)} data points using NV Simulator")
            return data
        except Exception as e:
            print(f"Error in NV simulation, using fallback: {e}")

    # Fallback: Load original dummy data or generate random data
    try:
        import os
        # Try multiple possible paths for the dummy data
        possible_paths = [
            os.path.join(os.path.dirname(__file__), 'FastComTec_demo_timetrace.asc'),
            os.path.expanduser('~/Desktop/KIT/SS25/DiamondBaby/QUSIM/mock_server/hardware/FastComTec_demo_timetrace.asc'),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                data = np.loadtxt(path, dtype='int64').tolist()
                print(f"Using dummy data from {path}")
                return data

        # If no dummy file found, generate random data
        print("No dummy data file found, generating random data")
        data = np.random.poisson(1.2, 1000).tolist()  # Poisson distributed counts
        return data

    except Exception as e:
        print(f"Error loading dummy data: {e}, generating minimal fallback")
        return [1, 0, 2, 1, 0, 1] * 100  # Simple repeating pattern