from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Generator
import os
import simpy
from environment import Environment, TimeSlot
from homeModel import Home, Room
from deviceModels import SmartLight
from inhabitantModel import Inhabitant

@dataclass
class TimeSlotState:
    StartTime: TimeSlot
    DayOfWeek: int                              # 1: Monday, 7: Sunday
    InhabitantStartLocation: Optional[int]      # Number of the location where inhabitant was at the start of the time slot
    DevicesStates: List[bool]                   # Devices on(True)/off(False) states
    InhabitantActions: List[Tuple[int, int]]    # Inhabitant actions (device number, action number) in current timeslot

    def __str__(self):
        return f'{self.StartTime.Minute},{self.StartTime.Hour},{self.DayOfWeek},{self.StartTime.Day},{self.StartTime.Month},{self.StartTime.Year},{self.InhabitantStartLocation},{",".join([("1" if state else "0") for state in self.DevicesStates])},[{";".join([f"{action[0]}:{action[1]}" for action in self.InhabitantActions])}]'


class StateLogger:
    '''Logs the state of the environment and location of given inhabitant in given time interval (time slot length).
    Logs are saved in a csv file.
    If the file already exists, logs will be appended without a header.

    Logs are saved in the following format:
    Minute,Hour,Day,Month,Year,Location,Device1_on,Device2_on,...,InhabitantActions[device_number:action_number;...;device_number:action_number]
    - Location: number of the room in the list of rooms in the home (starts from 1)
    - device_number: number of the device in the list of devices being logged (starts from 1)
    - action_number: number of the action the inhabitant performed with the device (starts from 1)'''
    def __init__(self, env: Environment, timeInterval: float, logFilePath: str, inhabitant: Inhabitant):
        self.env = env
        self.timeInterval = timeInterval
        
        self.logFilePath = logFilePath
        self.logFile = open(logFilePath, 'w')
        self.logFile.write('Minute,Hour,DayOfWeek,DayOfMonth,Month,Year,Location')

        self.state: TimeSlotState = TimeSlotState(env.timeslot, env.day_of_week(), None, [], []) # Current state

        # Location name to number mapping
        self.locationNumbers: Dict[str, int] = {}   # Room name, Room number
        # Names of devices being logged and their indices (order)
        self.deviceNumbers: Dict[str, int] = {} # Device name, Device number
        for room in self.env.home.rooms.values():
            self.locationNumbers[room.name] = len(self.locationNumbers) + 1
            for device in room.devices.values():
                if isinstance(device, SmartLight):  # Log only SmartLight devices # TODO: Log all devices
                    self.deviceNumbers[device.name] = len(self.deviceNumbers) + 1
                    self.state.DevicesStates.append(device.on)
                    self.logFile.write(',' + device.name + '_on')
        
        self.logFile.write(',InhabitantActions[...;...]\n')

        # Logged inhabitant reference
        self.inhabitant = inhabitant
        self.state.InhabitantStartLocation = self.locationNumbers[inhabitant.location.name] if inhabitant.location else None

    def log(self):
        # Log previous time slot
        self.logFile.write(str(self.state) + '\n')

        # Update current state
        self.state.StartTime = self.env.timeslot
        self.state.DayOfWeek = self.env.day_of_week()
        self.state.InhabitantStartLocation = self.locationNumbers[self.inhabitant.location.name] if self.inhabitant.location else None
        self.state.InhabitantActions = []
        for room in self.env.home.rooms.values():
            for device in room.devices.values():
                if isinstance(device, SmartLight) and device.name in self.deviceNumbers.keys(): # TODO: Log all devices
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
    # event "light_turned_on", self.name, inhabitant_name
    def deviceTurnedOnHandler(self, deviceName: str, inhabitantName: str):
        if self.inhabitant.name == inhabitantName:
            if deviceName in self.deviceNumbers.keys():
                self.state.InhabitantActions.append((self.deviceNumbers[deviceName], 2))
            else:
                print(f"Device {deviceName} turned ON, but is not being logged.")

    # event "light_turned_off", self.name, inhabitant_name
    def deviceTurnedOffHandler(self, deviceName: str, inhabitantName: str):
        if self.inhabitant.name == inhabitantName:
            if deviceName in self.deviceNumbers.keys():
                self.state.InhabitantActions.append((self.deviceNumbers[deviceName], 1))
            else:
                print(f"Device {deviceName} turned OFF, but is not being logged.")
    ################################

    def close(self):
        self.logFile.close()