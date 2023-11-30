from typing import Callable, Dict, Generator
from environment import TimeSlotEnvironment

class SmartDevice:
    def __init__(self, env: TimeSlotEnvironment, name: str) -> None:
        self.env = env
        self.name = name if name else "Unnamed"
        self.opMap: Dict[str, Callable] = {}

class SmartLight(SmartDevice):
    def __init__(self, env: TimeSlotEnvironment, name: str) -> None:
        super().__init__(env, name)
        self.on: bool = False
        self.opMap = {
            "turn_on": self.turn_on,
            "turn_off": self.turn_off,
            "remote_turn_on": self.remote_turn_on,
            "remote_turn_off": self.remote_turn_off
        }

    def turn_on(self) -> None:
        self.on = True
        print(f"? | -o {self.name} turned on")

    def turn_off(self) -> None:
        self.on = False
        print(f"? | -o {self.name} turned off")

    def remote_turn_on(self, inhabitant_name: str) -> None:
        self.on = True
        print(f"{inhabitant_name} | -o {self.name} turned on")

    def remote_turn_off(self, inhabitant_name: str) -> None:
        self.on = False
        print(f"{inhabitant_name} | -o {self.name} turned off")