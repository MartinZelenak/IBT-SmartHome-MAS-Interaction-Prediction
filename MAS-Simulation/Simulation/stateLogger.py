from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Generator, List, Optional, Tuple

import simpy

from .deviceModels import SmartLight
from .environment import Environment, Time
from .inhabitantModel import Inhabitant


@dataclass
class State:
    Timestamp: Time
    DayOfWeek: int                                  # 1: Monday, 7: Sunday
    InhabitantLocation: Optional[int]               # Number of the location where the inhabitant is
    DevicesStates: List[bool]                       # Devices on(True)/off(False) states
    InhabitantActions: List[Tuple[int, int]] | None # Inhabitant actions (device number, action number) in current timeslot. If None, won't print this column in string representation

    def __str__(self):
        if self.InhabitantActions is not None:
            return f'{self.Timestamp.Minute},{self.Timestamp.Hour},{self.DayOfWeek},{self.Timestamp.Day},{self.Timestamp.Month},{self.Timestamp.Year},{self.InhabitantLocation},{",".join([("1" if state else "0") for state in self.DevicesStates])},[{";".join([f"{action[0]}:{action[1]}" for action in self.InhabitantActions])}]'
        else:
            return f'{self.Timestamp.Minute},{self.Timestamp.Hour},{self.DayOfWeek},{self.Timestamp.Day},{self.Timestamp.Month},{self.Timestamp.Year},{self.InhabitantLocation},{",".join([("1" if state else "0") for state in self.DevicesStates])}'

class LogEvent(Enum):
    DEVICE_TURNED_ON = "DEVICE_TURNED_ON"
    DEVICE_TURNED_OFF = "DEVICE_TURNED_OFF"
    CHANGED_LOCATION = "CHANGED_LOCATION"

class StateLogger(ABC):
    def log(self):
        ...

    def deviceTurnedOnHandler(self):
        ...
    
    def deviceTurnedOffHandler(self):
        ...

    def inhabitantChangedLocation(self, inhabitantName: str, location_name: str):
        ...

    def close(self):
        ...

class PeriodicStateLogger(StateLogger):
    '''Logs the state of the environment and location of given inhabitant in given time interval (time slot length).
    Logs are saved in a csv file.
    If the file already exists, it will be overwritten.

    Logs are saved in the following format:
    Minute,Hour,DayOfWeek,Day,Month,Year,Location,Device1_on,Device2_on,...,InhabitantActions[device_number:action_number;...;device_number:action_number]
    - Location: number of the room in the list of rooms in the home (starts from 1)
    - device_number: number of the device in the list of devices being logged (starts from 1)
    - action_number: number of the action the inhabitant performed with the device (starts from 1)'''
    def __init__(self, env: Environment, timeInterval: float, logFilePath: str, inhabitant: Inhabitant):
        self.env = env
        self.timeInterval = timeInterval
        
        self.logFilePath = logFilePath
        self.logFile = open(logFilePath, 'w')
        self.logFile.write('Minute,Hour,DayOfWeek,DayOfMonth,Month,Year,Location')

        self.state: State = State(env.timeslot, env.day_of_week(), None, [], []) # Current state

        # Location name to number mapping
        self.locationNumbers: Dict[str, int] = {}   # Room name, Room number
        # Names of devices being logged and their indices (order)
        self.deviceNumbers: Dict[str, int] = {} # Device name, Device number
        for room in self.env.home.rooms.values():
            self.locationNumbers[room.name] = len(self.locationNumbers) + 1
            for device in room.devices.values():
                if isinstance(device, SmartLight):  # Log only SmartLight devices
                    self.deviceNumbers[device.name] = len(self.deviceNumbers) + 1
                    self.state.DevicesStates.append(device.on)
                    self.logFile.write(f',{device.name}_on')

        self.logFile.write(',InhabitantActions[...;...]\n')

        # Logged inhabitant reference
        self.inhabitant = inhabitant
        self.state.InhabitantLocation = self.locationNumbers[inhabitant.location.name] if inhabitant.location else None

    def log(self):
        # Log previous time slot
        self.logFile.write(f'{str(self.state)}\n')

        # Update current state
        self.state.Timestamp = self.env.timeslot
        self.state.DayOfWeek = self.env.day_of_week()
        self.state.InhabitantLocation = self.locationNumbers[self.inhabitant.location.name] if self.inhabitant.location else None
        self.state.InhabitantActions = []
        for room in self.env.home.rooms.values():
            for device in room.devices.values():
                if isinstance(device, SmartLight) and device.name in self.deviceNumbers.keys():
                    self.state.DevicesStates[self.deviceNumbers[device.name] - 1] = device.on

    # Simpy process
    # Periodic logging
    def logBehavior(self) -> Generator[simpy.Event, None, None]:
        while True:
            # Wait for next time slot and log the previous one
            yield self.env.timeout(self.timeInterval)
            self.log()

    # Actions during time slot
    ################################
    # event "light_turned_on", device_name, inhabitant_name
    def deviceTurnedOnHandler(self, deviceName: str, inhabitantName: str):
        if self.inhabitant.name == inhabitantName:
            if deviceName in self.deviceNumbers.keys():
                self.state.InhabitantActions.append((self.deviceNumbers[deviceName], 2))
            else:
                print(f"Device {deviceName} turned ON, but is not being logged.")

    # event "light_turned_off", device_name, inhabitant_name
    def deviceTurnedOffHandler(self, deviceName: str, inhabitantName: str):
        if self.inhabitant.name == inhabitantName:
            if deviceName in self.deviceNumbers.keys():
                self.state.InhabitantActions.append((self.deviceNumbers[deviceName], 1))
            else:
                print(f"Device {deviceName} turned OFF, but is not being logged.")
    
    # event "inhabitant_changed_location", inhabitant_name, location_name
    def inhabitantChangedLocation(self, inhabitantName: str, location_name: str):
        pass
    ################################

    def close(self):
        if not self.logFile.closed:
            self.logFile.close()

    def __del__(self):
        self.close()

class EventStateLogger(StateLogger):
    '''Logs the state of the environment and location of given inhabitant every time an event occures.
    Logs are saved in a csv file.
    If the file already exists it will be overwritten.

    Logs are saved in the following format:
    Minute,Hour,DayOfWeek,Day,Month,Year,Location,Device1_on,Device2_on,...
    - Location: number of the room in the list of rooms in the home (starts from 1)
    - device_number: number of the device in the list of devices being logged (starts from 1)
    - action_number: number of the action the inhabitant performed with the device (starts from 1)'''
    def __init__(self, env: Environment, logFilePath: str, inhabitant: Inhabitant):
        self.env = env

        self.logFilePath = logFilePath
        self.logFile = open(logFilePath, 'w')
        self.logFile.write('Minute,Hour,DayOfWeek,DayOfMonth,Month,Year,Location')

        self.state: State = State(env.timeslot, env.day_of_week(), None, [], None) # Current state

        self.last_log_time: Time | None = None

        # Location name to number mapping
        self.locationNumbers: Dict[str, int] = {}   # Room name, Room number
        # Names of devices being logged and their indices (order)
        self.deviceNumbers: Dict[str, int] = {} # Device name, Device number
        for room in self.env.home.rooms.values():
            self.locationNumbers[room.name] = len(self.locationNumbers) + 1
            for device in room.devices.values():
                if isinstance(device, SmartLight):  # Log only SmartLight devices
                    self.deviceNumbers[device.name] = len(self.deviceNumbers) + 1
                    self.state.DevicesStates.append(device.on)
                    self.logFile.write(f',{device.name}_on')

        self.logFile.write(',Events')

        # Logged inhabitant reference
        self.inhabitant = inhabitant
        self.state.InhabitantLocation = self.locationNumbers[inhabitant.location.name] if inhabitant.location else None

    def _updateState(self):
        self.last_log_time = self.state.Timestamp

        self.state.Timestamp = self.env.timeslot
        self.state.DayOfWeek = self.env.day_of_week()
        self.state.InhabitantLocation = self.locationNumbers[self.inhabitant.location.name] if self.inhabitant.location else None
        for room in self.env.home.rooms.values():
            for device in room.devices.values():
                if isinstance(device, SmartLight) and device.name in self.deviceNumbers.keys():
                    self.state.DevicesStates[self.deviceNumbers[device.name] - 1] = device.on

    def log(self, event_name: LogEvent):
        # Check if multiple events happened at the same time
        if self.last_log_time and self.last_log_time == self.state.Timestamp:
            ## just append another event name
            self.logFile.write(f'|{event_name.value}')
            return
        # Log current state
        self.logFile.write(f'\n{str(self.state)},{event_name.value}')
        self.last_log_time = self.state.Timestamp

    # event "light_turned_on", self.name, inhabitant_name
    def deviceTurnedOnHandler(self, deviceName: str, inhabitantName: str):
        if self.inhabitant.name == inhabitantName:
            if deviceName in self.deviceNumbers.keys():
                self._updateState()
                self.log(LogEvent.DEVICE_TURNED_ON)
            else:
                print(f"Device {deviceName} turned ON, but is not being logged.")

    # event "light_turned_off", self.name, inhabitant_name
    def deviceTurnedOffHandler(self, deviceName: str, inhabitantName: str):
        if self.inhabitant.name == inhabitantName:
            if deviceName in self.deviceNumbers.keys():
                self._updateState()
                self.log(LogEvent.DEVICE_TURNED_OFF)
            else:
                print(f"Device {deviceName} turned OFF, but is not being logged.")

    # event "user_changed_location", inhabitant_name, location_name
    def inhabitantChangedLocation(self, inhabitantName: str, location_name: str):
        if self.inhabitant.name == inhabitantName:
            self._updateState()
            self.state.InhabitantLocation = self.locationNumbers[location_name]
            self.log(LogEvent.CHANGED_LOCATION)

    def close(self):
        if not self.logFile.closed:
            self.logFile.write('\n')
            self.logFile.close()

    def __del__(self):
        self.close()
