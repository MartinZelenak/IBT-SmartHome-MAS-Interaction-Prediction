from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Generator
import simpy
from environment import Environment, TimeSlot
from homeModel import Home, Room
from deviceModels import SmartLight
from inhabitantModel import Inhabitant

@dataclass
class TimeSlotState:
    StartTime: TimeSlot
    InhabitantStartLocation: Optional[int]      # Number of the location where inhabitant was at the start of the time slot
    DevicesStates: List[bool]                   # Devices on(True)/off(False) states
    InhabitantActions: List[Tuple[int, int]]    # Inhabitant actions (device number, action number) in current timeslot

    def __str__(self):
        return f'{self.StartTime.Minute},{self.StartTime.Hour},{self.StartTime.Day},{self.StartTime.Month},{self.StartTime.Year},{self.InhabitantStartLocation},{",".join([("1" if state else "0") for state in self.DevicesStates])},[{";".join([f"{action[0]}:{action[1]}" for action in self.InhabitantActions])}]'


class StateLogger:
    '''Logs the state of the environment and location of given inhabitant in given time interval (time slot lenght).'''
    def __init__(self, env: Environment, timeInterval: float, logFilePath: str, inhabitant: Inhabitant):
        self.env = env
        self.timeInterval = timeInterval
        
        self.logFilePath = logFilePath
        self.logFile = open(logFilePath, 'w')
        self.logFile.write('Minute,Hour,Day,Month,Year,Location')

        self.state: TimeSlotState = TimeSlotState(env.timeslot, None, [], []) # Current state

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
        self.state.InhabitantStartLocation = self.locationNumbers[self.inhabitant.location.name] if self.inhabitant.location else None
        self.state.InhabitantActions = []
        for room in self.env.home.rooms.values():
            for device in room.devices.values():
                if isinstance(device, SmartLight) and device in self.deviceNumbers.keys(): # TODO: Log all devices
                    self.state.DevicesStates[self.deviceNumbers[device.name] - 1] = device.on

    # Simpy process
    # Periodic logging
    def logBehavior(self) -> Generator[simpy.Event, None, None]:
        while True:
            self.log()
            yield self.env.timeout(self.timeInterval)

    # Actions during time slot
    ################################
    # event "light_turned_on", self.name, inhabitant_name
    def deviceTurnedOnHandler(self, deviceName: str, inhabitantName: str):
        if self.inhabitant.name == inhabitantName:
            if deviceName in self.deviceNumbers.keys():
                self.state.InhabitantActions.append((self.deviceNumbers[deviceName], 1))
            else:
                print(f"Device {deviceName} turned ON, but is not being logged.")

    # event "light_turned_off", self.name, inhabitant_name
    def deviceTurnedOffHandler(self, deviceName: str, inhabitantName: str):
        if self.inhabitant.name == inhabitantName:
            if deviceName in self.deviceNumbers.keys():
                self.state.InhabitantActions.append((self.deviceNumbers[deviceName], 0))
            else:
                print(f"Device {deviceName} turned OFF, but is not being logged.")
    ################################

    def close(self):
        self.logFile.close()