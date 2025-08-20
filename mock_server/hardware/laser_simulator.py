from fastapi import APIRouter

from pydantic import BaseModel, Field
from enum import IntEnum
from typing import List, FrozenSet, Dict

import numpy as np


### --- Models for Laser Control --- ###
# - Enums from Interface class - #
class ControlMode(IntEnum):
    POWER = 0
    CURRENT = 1
    UNKNOWN = 2

class ShutterState(IntEnum):
    CLOSED = 0
    OPEN = 1
    NO_SHUTTER = 2
    UNKNOWN = 3

class LaserState(IntEnum):
    OFF = 0
    ON = 1
    LOCKED = 2
    UNKNOWN = 3

# - Pydantic Model for Laser Data - #
class LaserModel(BaseModel):
    power: float =                   Field(..., ge=0, description="Current laser power in watts")
    power_setpoint: float =          Field(..., ge=0, description="Desired laser power setpoint in watts")
    current: float =                 Field(..., ge=0, description="Current laser current")
    current_setpoint: float =        Field(..., ge=0, description="Desired laser current setpoint")
    control_mode: ControlMode =      Field(...,       description="Current control mode of the laser")
    shutter_state: ShutterState =    Field(...,       description="Current state of the laser shutter")
    laser_state: LaserState =        Field(...,       description="Current state of the laser")
    temperatures: Dict[str, float] = Field(...,       description="Temperatures of the laser components in degrees Celsius")


### --- FastAPI Application Setup --- ###
router = APIRouter()

# Initial State of the Laser
laser_data = LaserModel(
    power=0.0,
    power_setpoint=0.0,
    current=0.0,
    current_setpoint=0.0,
    control_mode=ControlMode.POWER,
    shutter_state=ShutterState.CLOSED,
    laser_state=LaserState.OFF,
    temperatures={"psu": 25.0, "head": 30.0}
)

@router.get("/laser/init", response_model=LaserModel)
def initialize_laser() -> LaserModel:
    """ Initialize the laser with default values. """
    return laser_data

@router.get("/laser/status", response_model=LaserModel)
def get_laser_status() -> LaserModel:
    """ Get the current status of the laser. """
    return laser_data

@router.get("/laser/on_activate", response_model=LaserModel)
def on_activate() -> LaserModel:
    """ Activate the laser and return its current state. """
    # Here you would typically perform activation logic
    return laser_data

@router.get("/laser/on_deactivate", response_model=LaserModel)
def on_deactivate() -> LaserModel:
    """ Deactivate the laser and return its current state. """
    # Here you would typically perform deactivation logic
    return laser_data

@router.get("/laser/power/range", response_model=List[float])
def get_power_range() -> List[float]:
    """ Return the power range of the laser in watts. """
    return [0.0, 0.250]

@router.get("/laser/power", response_model=float)
def get_power() -> float:
    """ Return the current power of the laser in watts. """
    # Simulate power reading with some noise
    laser_data.power = laser_data.power_setpoint * np.random.normal(1, 0.01)
    return laser_data.power

@router.get("/laser/power/setpoint", response_model=float)
def get_power_setpoint() -> float:
    """ Return the current power setpoint of the laser in watts. """
    return laser_data.power_setpoint

@router.post("/laser/power/setpoint", response_model=None)
def set_power_setpoint(power_setpoint: float):
    """ Set the power setpoint of the laser in watts. """
    # Validate the power setpoint against the range

    minPower, maxPower = get_power_range()
    minCurrent, maxCurrent = get_current_range()
    
    if power_setpoint < minPower or power_setpoint > maxPower:
        raise ValueError(f"Power setpoint must be between {minPower} and {maxPower} watts.")
    
    # Update the laser data with the new power setpoint
    # Update current setpoint based on power setpoint via current = sqrt(power)
    laser_data.power_setpoint = power_setpoint
    laser_data.current_setpoint = np.sqrt(laser_data.power_setpoint / maxPower) * maxCurrent

@router.get("/laser/current/unit", response_model=str)
def get_current_unit() -> str:
    """ Get the unit for laser current. """
    return "%"

@router.get("/laser/current/range", response_model=List[float])
def get_current_range() -> List[float]:
    """ Get the current range of the laser in laser current units. 
        Here we assume percents and return a range of 0 to 100.
    """
    return [0.0, 100.0]

@router.get("/laser/current", response_model=float)
def get_current() -> float:
    """ Get the actual laser current in laser current units. """
    # Simulate current reading with some noise
    laser_data.current = laser_data.current_setpoint * np.random.normal(1, 0.05)
    return laser_data.current

@router.get("/laser/current/setpoint", response_model=float)
def get_current_setpoint() -> float:
    """ Get the current current setpoint of the laser in laser current units. """
    return laser_data.current_setpoint

@router.post("/laser/current/setpoint", response_model=None)
def set_current_setpoint(current_setpoint: float):
    """ Set the current setpoint of the laser in laser current units. """
    # Validate the current setpoint against the range
    minPower, maxPower = get_power_range()
    minCurrent, maxCurrent = get_current_range()
    if current_setpoint < minCurrent or current_setpoint > maxCurrent:
        raise ValueError(f"Current setpoint must be between {minCurrent} and {maxCurrent} laser current units.")

    # Update the laser data with the new current setpoint
    laser_data.current_setpoint = current_setpoint
    laser_data.power_setpoint = np.power(current_setpoint / 100, 2) * maxPower

@router.get("/laser/control_mode/allowed", response_model=FrozenSet[ControlMode])
def allowed_control_modes() -> FrozenSet[ControlMode]:
    """ Get supported control modes for the laser. """
    return frozenset({ControlMode.POWER, ControlMode.CURRENT})

@router.get("/laser/control_mode", response_model=ControlMode)
def get_control_mode() -> ControlMode:
    """ Get the currently active control mode of the laser. """
    return laser_data.control_mode

@router.post("/laser/control_mode", response_model=None)
def set_control_mode(control_mode: ControlMode):
    """ Set the active control mode of the laser. """
    if control_mode not in allowed_control_modes():
        raise ValueError(f"Control mode {control_mode} is not supported.")
    
    laser_data.control_mode = control_mode

@router.post("/laser/on", response_model=None)
def on():
    """ Turn on the laser and return its current state. """
    # Here you would typically perform the logic to turn on the laser
    laser_data.laser_state = LaserState.ON

@router.post("/laser/off", response_model=None)
def off():
    """ Turn off the laser and return its current state. """
    # Here you would typically perform the logic to turn off the laser
    laser_data.laser_state = LaserState.OFF

@router.get("/laser/state", response_model=LaserState)
def get_laser_state() -> LaserState:
    """ Get the current state of the laser. """
    return laser_data.laser_state

@router.post("/laser/state", response_model=None)
def set_laser_state(state: LaserState):
    """ Set the laser state. """
    if state not in {LaserState.ON, LaserState.OFF, LaserState.LOCKED}:
        raise ValueError(f"Laser state {state} is not supported.")
    
    laser_data.laser_state = state

@router.get("/laser/shutter/state", response_model=ShutterState)
def get_shutter_state():
    """ Get the current state of the laser shutter. """
    return laser_data.shutter_state

@router.post("/laser/shutter/state", response_model=None)
def set_shutter_state(state: ShutterState):
    """ Set the laser shutter state. """    
    if state not in {ShutterState.CLOSED, ShutterState.OPEN, ShutterState.NO_SHUTTER}:
        raise ValueError(f"Shutter state {state} is not supported.")
    
    laser_data.shutter_state = state
    
@router.get("/laser/temperatures", response_model=Dict[str, float])
def get_temperatures() -> Dict[str, float]:
    """ Get all available temperatures of the laser. """
    # Simulate temperature readings with some noise
    laser_data.temperatures = {
            'psu':  25.0 * np.random.normal(1, 0.1),
            'head': 30.0 * np.random.normal(1, 0.2)
        }
    return laser_data.temperatures

@router.get("/laser/extra_info", response_model=str)
def get_extra_info() -> str:
    """ Get multiple lines of diagnostic information about the laser. """
    return "Laser API v1.0\nThis is a mock laser simulator.\nUse with care."