from typing import Callable, Dict

from .environment import Environment


class SmartDevice:
    def __init__(self, env: Environment, name: str) -> None:
        self.env = env
        self.name = name or "Unnamed"
        self.opMap: Dict[str, Callable] = {}
        
        # MAS stats
        self.MAS_changed_state: bool = False
        self.MAS_correct_actions: int = 0
        self.MAS_incorrect_actions: int = 0

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

        self.opMap["turn_on"] = self.turn_on
        self.opMap["turn_off"] = self.turn_off
    
    def turn_on(self, inhabitant_name: str) -> bool:
        if self.on:
            if self.MAS_changed_state:
                self.MAS_correct_actions += 1
                self.MAS_changed_state = False
            return False
        self.on = True
        if self.MAS_changed_state:
            self.MAS_incorrect_actions += 1
            self.MAS_changed_state = False

        self.env.eventHandler.publish("light_turned_on", self.name, inhabitant_name)
        print(f"{inhabitant_name} | -o {self.name} turned on")
        return True

    def turn_off(self, inhabitant_name: str) -> bool:
        if not self.on:
            if self.MAS_changed_state:
                self.MAS_correct_actions += 1
                self.MAS_changed_state = False
            return False
        self.on = False
        if self.MAS_changed_state:
            self.MAS_incorrect_actions += 1
            self.MAS_changed_state = False

        self.env.eventHandler.publish("light_turned_off", self.name, inhabitant_name)
        print(f"{inhabitant_name} | -o {self.name} turned off")
        return True

    def MAS_turn_on(self) -> bool:
        if self.on:
            return False
        self.on = True
        self.state_changed_by_MAS = True
        print(f"MAS | -o {self.name} turned on")
        return True

    def MAS_turn_off(self) -> bool:
        if not self.on:
            return False
        self.on = False
        self.state_changed_by_MAS = True
        print(f"MAS | -o {self.name} turned off")
        return True

SmartTV = SmartLight