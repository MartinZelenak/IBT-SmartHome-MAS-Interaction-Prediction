from dataclasses import dataclass
from typing import Dict, Optional
import json

@dataclass()
class TimeSlot:
    '''The time of the environment.
    This class represents the time of the environment, that is passed to the system.'''
    Minute: int
    Hour: int
    DayOfWeek: int

    def __init__(self, minute: int, hour: int, dayOfWeek: int):
        if minute < 0 or minute > 59:
            raise ValueError('Invalid minute')
        self.Minute = minute

        if hour < 0 or hour > 23:
            raise ValueError('Invalid hour')
        self.Hour = hour

        if dayOfWeek < 0 or dayOfWeek > 6:
            raise ValueError('Invalid day of week')
        self.DayOfWeek = dayOfWeek

    def __len__(self):
        return 3

@dataclass()
class State:
    '''The state of the environment.
    This class represents the state of the environment, that is passed to the system.
    
    Attributes:
        Time: The time of the environment.
        UserLocations: A dictionary of user JIDs and their location ids (id >= 0).
        DeviceStates: A dictionary of device agent JIDs and their state values.'''
    # UserLocations
    @property
    def UserLocations(self) -> Dict[str, int]:
        return self._userLocations
    @UserLocations.setter
    def UserLocations(self, value: Dict[str, int]):
        if not isinstance(value, dict):
            raise ValueError('Invalid user locations')
        for user_jid, location_id in value.items():
            if not isinstance(user_jid, str):
                raise ValueError('Invalid user JID')
            if not isinstance(location_id, int) or location_id < 0:
                raise ValueError('Invalid location ID')
        self._userLocations = value

    # DeviceStates
    @property
    def DeviceStates(self) -> Dict[str, int|float]:
        return self._deviceStates
    @DeviceStates.setter
    def DeviceStates(self, value: Dict[str, int|float]):
        if not isinstance(value, dict):
            raise ValueError('Invalid device states')
        for device_jid, device_state in value.items():
            if not isinstance(device_jid, str):
                raise ValueError('Invalid device JID')
            if not isinstance(device_state, int) and not isinstance(device_state, float):
                raise ValueError('Invalid state value')
        self._deviceStates = value


    def __init__(self, user_locations: Optional[Dict[str, int]] = None, device_states: Optional[Dict[str, int|float]] = None):
        self.UserLocations = user_locations if user_locations is not None else {}
        self.DeviceStates = device_states if device_states is not None else {}
    
    def __str__(self):
        '''Converts the State object to a string representation.'''
        user_locations_str = ', '.join([f'{jid}: {location_id}' for jid, location_id in self.UserLocations.items()])
        device_states_str = ', '.join([f'{jid}: {state}' for jid, state in self.DeviceStates.items()])
        return f'State(User Locations: {user_locations_str}, Device States: {device_states_str})'
    
    def to_json(self) -> str:
        '''Converts the State object to a JSON representation.'''
        state_dict = {
            'UserLocations': self.UserLocations,
            'DeviceStates': self.DeviceStates
        }
        return json.dumps(state_dict)

    @staticmethod
    def from_json(json_str: str) -> 'State':
        '''Converts a JSON representation to a State object.'''
        state_dict = json.loads(json_str)
        return State(state_dict['UserLocations'], state_dict['DeviceStates'])

@dataclass()
class DeviceFilter:
    # TODO: Enable/disable only sometimes: time_from, time_to, days
    Device_JID: str
    Enabled: Optional[bool]
    Treshold_Off: Optional[float]
    Treshold_On: Optional[float]

    def to_json(self) -> str:
        device_filter_dict = {
            'Device_JID': self.Device_JID,
            'Enabled': self.Enabled,
            'Treshold_Off': self.Treshold_Off,
            'Treshold_On': self.Treshold_On
        }
        return json.dumps(device_filter_dict)
    
    @staticmethod
    def from_json(json_str: str) -> 'DeviceFilter':
        json_dict = json.loads(json_str)
        return DeviceFilter(json_dict['Device_JID'], json_dict['Enabled'], json_dict['Treshold_Off'], json_dict['Treshold_On'])