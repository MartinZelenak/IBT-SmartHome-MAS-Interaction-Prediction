"""
Author: Martin Zelenák (xzelen27@stud.fit.vutbr.cz)
Description: A simple event handler for subscribing to and publishing events in the simulation.
Date: 2025-05-14
"""


from typing import Dict, List, Callable

class EventHandler:
    def __init__(self) -> None:
        self.subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_name: str, callback: Callable) -> None:
        if event_name not in self.subscribers:
            self.subscribers[event_name] = []
        self.subscribers[event_name].append(callback)

    def publish(self, event_name: str, *args, **kwargs) -> None:
        if event_name not in self.subscribers:
            return
        for callback in self.subscribers[event_name]:
            callback(*args, **kwargs)