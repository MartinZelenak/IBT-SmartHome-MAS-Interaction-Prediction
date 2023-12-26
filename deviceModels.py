from typing import Callable, Dict, Generator
from environment import Environment

class SmartDevice:
    def __init__(self, env: Environment, name: str) -> None:
        self.env = env
        self.name = name if name else "Unnamed"
        self.opMap: Dict[str, Callable] = {}

class SmartLight(SmartDevice):
    '''A smart light device.
    State:
        - on: bool
    Operations:
        - turn_on(inhabitant_name: str)
        - turn_off(inhabitant_name: str)
    '''
    def __init__(self, env: Environment, name: str) -> None:
        super().__init__(env, name)
        self.on: bool = False
        self.opMap = {
            "turn_on": self.turn_on,
            "turn_off": self.turn_off
        }
    
    def turn_on(self, inhabitant_name: str) -> None:
        self.on = True
        self.env.eventHandler.publish("light_turned_on", self.name, inhabitant_name)
        print(f"{inhabitant_name} | -o {self.name} turned on")

    def turn_off(self, inhabitant_name: str) -> None:
        self.on = False
        self.env.eventHandler.publish("light_turned_off", self.name, inhabitant_name)
        print(f"{inhabitant_name} | -o {self.name} turned off")