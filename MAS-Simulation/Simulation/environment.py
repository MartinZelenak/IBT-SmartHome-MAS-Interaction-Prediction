"""
Author: Martin Zelenák (xzelen27@stud.fit.vutbr.cz)
Description: The Environment and Time classes for simulating time, scheduling, and event handling in the smart home simulation.
Date: 2025-05-14
"""


from typing import Any, Callable, List, NamedTuple, Optional

import simpy

from .event import EventHandler


class Time(NamedTuple):
    Minute: float
    Hour: int
    Day: int
    Month: int
    Year: int

    def day_of_week(self) -> int:
        total_days = (self.Year - 1) * 365 + (self.Month - 1) * 31 + (self.Day - 1)
        return (total_days % 7) + 1

    def __str__(self):
        return f'{self.Hour:02}:{int(self.Minute):02} {(self.Day):02}.{self.Month:02}.{self.Year:04}'

    def __add__(self, other: 'Time | float | int'):
        if isinstance(other, (float, int)):
            other = Time.from_minutes(other)

        minute = self.Minute + other.Minute
        hour = self.Hour + other.Hour
        day = self.Day + other.Day
        month = self.Month + other.Month
        year = self.Year + other.Year

        if minute >= 60:
            minute -= 60
            hour += 1

        if hour >= 24:
            hour -= 24
            day += 1

        if day > 31:
            day -= 31
            month += 1

        if month > 12:
            month -= 12
            year += 1

        return Time(minute, hour, day, month, year)

    def __sub__(self, other: 'Time | float | int'):
        if isinstance(other, (float, int)):
            other = Time.from_minutes(other)

        minute = self.Minute - other.Minute
        hour = self.Hour - other.Hour
        day = self.Day - other.Day
        month = self.Month - other.Month
        year = self.Year - other.Year

        if minute < 0:
            minute += 60
            hour -= 1

        if hour < 0:
            hour += 24
            day -= 1

        if day < 1:
            day += 31
            month -= 1

        if month < 1:
            month += 12
            year -= 1

        return Time(minute, hour, day, month, year)

    def __lt__(self, other: 'Time | float | int'):
        if isinstance(other, (float, int)):
            other = Time.from_minutes(other)
        return self.to_minutes() < other.to_minutes()

    def __gt__(self, other: 'Time | float | int'):
        if isinstance(other, (float, int)):
            other = Time.from_minutes(other)
        return self.to_minutes() > other.to_minutes()

    def __eq__(self, other: 'Time | float | int'):
        if isinstance(other, (float, int)):
            other = Time.from_minutes(other)
        return self.to_minutes() == other.to_minutes()

    def to_minutes(self):
        return self.Minute + self.Hour * 60 + (self.Day - 1) * 24 * 60 + (self.Month - 1) * 31 * 24 * 60 + (self.Year - 1) * 12 * 31 * 24 * 60
    
    @staticmethod
    def from_minutes(min: float | int) -> 'Time':
        return Time(
                Minute = min % 60,
                Hour = int((min // 60) % 24), 
                Day = int((min // (60*24)) % 31 + 1), 
                Month = int((min // (60*24*31)) % 12 + 1), 
                Year = int((min // (60*24*31*12)) + 1)
        )	

class TimeoutRequest(simpy.Event):
    '''Functions just as simpy.Timeout, but is scheduled/triggered only after confirming
    '''
    def __init__(
        self,
        env: simpy.Environment,
        delay: int | float,
        value: Optional[Any] = None,
    ):
        # NOTE: This code is taken from the simpy.Timeout.__init__() method
        if delay < 0:
            raise ValueError(f'Negative timeout delay {delay}')
        self.env = env
        self.callbacks: List[Callable[[simpy.Event], None]] = []
        self._value = value
        self._delay = delay
        self._ok = True

    def confirm(self):
        '''Schedules the timeout (puts it into event queue)
        '''
        self.env.schedule(self, delay = self._delay)

def get_home():
    '''Returns the Home class from homeModel.py. Needed to avoid circular imports.'''
    from .homeModel import Home
    return Home

class Environment(simpy.Environment):
    def __init__(self, initial_time = 0):
        super().__init__(initial_time)
        self.home = get_home()(self)
        self.inhabitans = []
        self.eventHandler = EventHandler()

    @property
    def timeslot(self) -> Time:
        return Time.from_minutes(self.now)

    def day_of_week(self) -> int:
        return self.timeslot.day_of_week()

    def is_weekend(self) -> bool:
        return self.timeslot.day_of_week() > 5
    
    def timeoutUntil(self, time: Time) -> simpy.Event:
        return self.timeout(time.to_minutes() - self.now)
    
    def timeoutRequest(self, delay: float | int, value: Optional[Any] = None) -> simpy.Event:
        return TimeoutRequest(self, delay, value)
