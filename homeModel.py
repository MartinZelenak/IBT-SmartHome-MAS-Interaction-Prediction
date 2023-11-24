from environment import TimeSlotEnvironment

class SmartDevice:
    def __init__(self, env: TimeSlotEnvironment, name: str) -> None:
        self.env = env
        self.name = name if name else "Unnamed"


class Room:
    def __init__(self, env: TimeSlotEnvironment, name: str, temperature: float = 20) -> None:
        self.env = env
        self.name = name if name else "Unnamed"
        self.temperature = temperature  # Celsius
        self.devices = []

    def add_device(self, device: SmartDevice) -> None:
        self.devices.append(device)


class Home:
    def __init__(self, env: TimeSlotEnvironment) -> None:
        self.env = env
        self.rooms = {}

    def add_room(self, room: Room) -> None:
        if room.name in self.rooms:
            raise ValueError(f"Home already has a room called {room.name}!")
        self.rooms[room.name] = room

    def go_to_room(self, room_name: str) -> Room:
        if room_name not in self.rooms:
            raise ValueError(f"Home does not have a room called {room_name}!")
        # TODO: log location (needs inhabitant name/id)
        return self.rooms[room_name]
    
    def get_room(self, room_name: str) -> Room:
        if room_name not in self.rooms:
            raise ValueError(f"Home does not have a room called {room_name}!")
        # Doesn't log location (used for remote control)
        return self.rooms[room_name]