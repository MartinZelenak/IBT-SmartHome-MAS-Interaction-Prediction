"""
Author: Martin Zelenák (xzelen27@stud.fit.vutbr.cz)
Description: The DeterministicInhabitant class modeling deterministic inhabitant
            behavior and state transitions for the smart home simulation.
Date: 2025-05-14
"""


import random
from typing import Generator, override

import simpy

from .. import inhabitantModel as im
from ..environment import Environment


class DeterministicInhabitant(im.Inhabitant):
    def __init__(self, env: Environment, name: str, weekend_same_as_workday_behavior: bool = False) -> None:
        super().__init__(env, name, weekend_same_as_workday_behavior)
        self.deterministic = True

    @override
    def sleeps_state(self) -> Generator[simpy.Event, None, None]:
        '''Sleeps state must be prolonged'''
        # Go to the bedroom
        if(not self.is_in_room('bedroom')):
            yield self.env.timeoutRequest(1) # 1 minute
            self.change_room('bedroom')

        ## Turns the lights off
        self.location.get_device_op('bedroom_light', 'turn_off')(self.name)

        yield self.env.timeoutRequest(0)

    @override
    def wakes_up_state(self) -> Generator[simpy.Event, None, None]:
        if(self.env.is_weekend()):
            # Doesn't get up immediately
            yield self.env.timeoutRequest(7.5) # 7.5 minutes

            ## Turns the lights on
            self.location.get_device_op('bedroom_light', 'turn_on')(self.name)

            # Put on home clothes
            yield self.env.timeoutRequest(3.5) # 3.5 minutes

            ## Turns the lights off
            self.location.get_device_op('bedroom_light', 'turn_off')(self.name)

        # Go to the bathroom
        if(not self.is_in_room('bathroom')):
            yield self.env.timeoutRequest(0.3) # 0.3 minutes
            self.change_room('bathroom')

        ## Turns the lights on
        self.location.get_device_op('bathroom_light', 'turn_on')(self.name)
        
        # Brush teeth
        yield self.env.timeoutRequest(2.5) # 2.5 minutes

        ## Turns the lights off
        self.location.get_device_op('bathroom_light', 'turn_off')(self.name)

        # Go to the kitchen
        yield self.env.timeoutRequest(0.75) # 0.75 minutes
        self.change_room('kitchen')

        # Make coffee
        yield self.env.timeoutRequest(3.5) # 3.5 minutes

        
    @override
    def prepares_to_leave_state(self) -> Generator[simpy.Event, None, None]:
        # Go to the bedroom
        if(not self.is_in_room('bedroom')):
            yield self.env.timeoutRequest(0.75) # 0.75 minutes
            self.change_room('bedroom')

        ## Turns the lights on
        lightIsOn = False
        if(self.env.timeslot.Hour < 8 or self.env.timeslot.Hour >= 21):
            self.location.get_device_op('bedroom_light', 'turn_on')(self.name)
            lightIsOn = True

        # Put on clothes and grab stuff
        yield self.env.timeoutRequest(5) # 5 minutes

        ## Turns the lights off
        if(lightIsOn):
            self.location.get_device_op('bedroom_light', 'turn_off')(self.name)

        # Go to the door
        yield self.env.timeoutRequest(2.5) # 2.5 minutes
        self.change_room('hallway')

        # Put on shoes
        yield self.env.timeoutRequest(1.5) # 1.5 minutes

    @override
    def left_state(self) -> Generator[simpy.Event, None, None]:
        '''Left state must be prolonged'''
        yield self.env.timeoutRequest(0.1)
        self.change_room('outside')
        yield self.env.timeoutRequest(0)

    @override
    def arrives_state(self) -> Generator[simpy.Event, None, None]:
        # Enter the hallway
        self.change_room('hallway')
        
        # Put off shoes
        yield self.env.timeoutRequest(0.75) # 0.75 minutes

        # Go to the bedroom
        yield self.env.timeoutRequest(0.75) # 0.75 minutes
        self.change_room('bedroom')

        # Put off clothes and put on home clothes
        yield self.env.timeoutRequest(5) # 5 minutes

    @override
    def relaxes_state(self) -> Generator[simpy.Event, None, None]:
        '''Relaxes state must be cut short'''
        # Go to the living room
        if(not self.is_in_room('livingroom')):
            yield self.env.timeoutRequest(0.75)
            self.change_room('livingroom')

        # Sit on the couch
        yield self.env.timeoutRequest(0.75) # 0.75 minutes
        
        ## Turn on the TV
        self.location.get_device_op('livingroom_tv', 'turn_on')(self.name)

        # Watch TV
        try:
            while True:
                # Will be cut short by stateEnd.max
                yield self.env.timeoutRequest(15)
        finally:
            ## Turn off the TV
            self.location.get_device_op('livingroom_tv', 'turn_off')(self.name)

    @override
    def reads_state(self) -> Generator[simpy.Event, None, None]:
        '''Reads state must be cut short'''
        # Go to the livingroom
        if(not self.is_in_room('livingroom')):
            yield self.env.timeoutRequest(0.75)
            self.change_room('livingroom')

        ## Turn on the livingroom light
        self.location.get_device_op('livingroom_light', 'turn_on')(self.name)

        # Sit on the couch
        yield self.env.timeoutRequest(0.75)

        # Read a book
        try:
            while True:
                # Will be cut short by stateEnd.max
                yield self.env.timeoutRequest(12.5)
        finally:
            ## Turn off the livingroom light
            self.location.get_device_op('livingroom_light', 'turn_off')(self.name)

    @override
    def does_hobby_state(self) -> Generator[simpy.Event, None, None]:
        '''Scrolls through his/her phone in the bedroom'''
        # Go to the bedroom
        if(not self.is_in_room('bedroom')):
            yield self.env.timeoutRequest(0.75)
            self.change_room('bedroom')

        # Lay on the bed
        yield self.env.timeoutRequest(0.75)

        # Scroll through phone
        while True:
            # Will be cut short by stateEnd.max
            yield self.env.timeoutRequest(20)

    @override
    def works_state(self) -> Generator[simpy.Event, None, None]:
        # Go to the office
        if(not self.is_in_room('office')):
            yield self.env.timeoutRequest(0.75)
            self.change_room('office')

        ## Sometimes works with lights on
        if(random.random() < 0.8):
            self.location.get_device_op('office_light', 'turn_on')(self.name)

        # Sit on the chair
        yield self.env.timeoutRequest(0.35)

        # Work
        try:
            while True:
                # Will be cut short by stateEnd.max
                yield self.env.timeoutRequest(30)
        finally:
            ## Turn off the office light
            self.location.get_device_op('office_light', 'turn_off')(self.name)

    @override
    def prepares_food_state(self) -> Generator[simpy.Event, None, None]:
        # Go to the kitchen
        if(not self.is_in_room('kitchen')):
            yield self.env.timeoutRequest(0.75)
            self.change_room('kitchen')

        ## Turn on the kitchen light
        if self.env.timeslot.Hour > 19:
            self.location.get_device_op('kitchen_light', 'turn_on')(self.name)

        # Prepare food
        yield self.env.timeoutRequest(30)

        ## Turn off the kitchen light
        self.location.get_device_op('kitchen_light', 'turn_off')(self.name)

    @override
    def eats_state(self) -> Generator[simpy.Event, None, None]:
        # Go to the livingroom
        if(not self.is_in_room('livingroom')):
            yield self.env.timeoutRequest(0.75)
            self.change_room('livingroom')

        ## Turn on the livingroom light
        if self.env.timeslot.Hour > 19:
            self.location.get_device_op('livingroom_light', 'turn_on')(self.name)

        # Sit on a chair
        yield self.env.timeoutRequest(0.75)

        # Eat
        yield self.env.timeoutRequest(25)

        ## Turn off the livingroom light
        self.location.get_device_op('livingroom_light', 'turn_off')(self.name)


    @override
    def workday_behavior(self, currentState: im.InhabitantState) -> Generator[simpy.Event, None, None] | None:
        # Next state logic
        currentTimeslot = self.env.timeslot
        if(currentTimeslot.Hour < 6):
            # Sleeps until 6:00
            self.state = im.InhabitantState.SLEEPS
            end = currentTimeslot._replace(Hour = 6, Minute = 0).to_minutes()
            self.stateEnd = im.stateEnd(end, None)

        elif(currentTimeslot.Hour == 6):
            # Wakes up after Sleeping
            if(currentState == im.InhabitantState.SLEEPS):
                self.state = im.InhabitantState.WAKES_UP
            # Prepares to leave until 7:05
            elif(currentState == im.InhabitantState.WAKES_UP):
                self.state = im.InhabitantState.PREPARES_TO_LEAVE
                end = currentTimeslot._replace(Hour = 7, Minute = 5).to_minutes()
                self.stateEnd = im.stateEnd(end, end)
            else:
                # Leaves until 16:20
                self.state = im.InhabitantState.LEFT
                end = currentTimeslot._replace(Hour = 16, Minute = 20).to_minutes()
                self.stateEnd = im.stateEnd(end, end)
        
        elif(currentTimeslot.Hour >= 7 and currentTimeslot.Hour <= 15):
            # Leaves until 16:20
            self.state = im.InhabitantState.LEFT
            end = currentTimeslot._replace(Hour = 16, Minute = 20).to_minutes()
            self.stateEnd = im.stateEnd(end, end)

        elif(currentState == im.InhabitantState.LEFT):
            self.state = im.InhabitantState.ARRIVES
            
        elif(currentTimeslot.Hour == 16):
            # Relaxes until 17:25
            self.state = im.InhabitantState.RELAXES
            end = currentTimeslot._replace(Hour = 17, Minute = 25).to_minutes()
            self.stateEnd = im.stateEnd(end, end)
        
        elif(currentTimeslot.Hour >= 17 and currentTimeslot.Hour <= 19):
            # Does hobby until 20:15
            self.state = im.InhabitantState.DOES_HOBBY
            end = currentTimeslot._replace(Hour = 20, Minute = 15).to_minutes()
            self.stateEnd = im.stateEnd(end, end)

        elif(currentTimeslot.Hour == 20):
            # Reads until 21:30
            self.state = im.InhabitantState.READS
            end = currentTimeslot._replace(Hour = 21, Minute = 30).to_minutes()
            self.stateEnd = im.stateEnd(end, end)

        elif(currentTimeslot.Hour >= 21 and (currentTimeslot.Hour <= 22 and currentTimeslot.Minute <= 45)):
            # Prepares food and Eats until 22:45
            self.state = im.InhabitantState.PREPARES_FOOD
            if(currentState == im.InhabitantState.PREPARES_FOOD):
                self.state = im.InhabitantState.EATS
                end = currentTimeslot._replace(Hour = 22, Minute = 45).to_minutes()
                self.stateEnd = im.stateEnd(end, end)
        
        elif((currentTimeslot.Hour == 22 and currentTimeslot.Minute >= 45) or currentTimeslot.Hour >= 23):
            # Sleeps until 6:00 (next day)
            self.state = im.InhabitantState.SLEEPS
            end = currentTimeslot._replace(Hour = 6, Minute = 0).to_minutes() + 24*60
            self.stateEnd = im.stateEnd(end, None)

        # Current state
        if(currentState != self.state):
            print(f'{self.name} | Workday: {self.env.timeslot} - {self.state}')



    @override
    def weekend_behavior(self, currentState: im.InhabitantState) -> Generator[simpy.Event, None, None] | None:
        # Next state logic
        currentTimeslot = self.env.timeslot
        if(currentTimeslot.Hour < 8):
            # Sleeps until 8:00
            self.state = im.InhabitantState.SLEEPS
            end = currentTimeslot._replace(Hour = 8, Minute = 0).to_minutes()
            self.stateEnd = im.stateEnd(end, end)

        elif(currentTimeslot.Hour >= 8 and currentTimeslot.Hour <= 11):
            # Wakes up after Sleeping and Works at home until 12:00
            if(currentState == im.InhabitantState.SLEEPS):
                self.state = im.InhabitantState.WAKES_UP
            else:
                self.state = im.InhabitantState.WORKS

                end = currentTimeslot._replace(Hour = 12, Minute = 00).to_minutes()
                self.stateEnd = im.stateEnd(end, end)

        elif(currentTimeslot.Hour == 12):
            # Prepares food and eats until 13:15
            self.state = im.InhabitantState.PREPARES_FOOD
            if(currentState == im.InhabitantState.PREPARES_FOOD):
                self.state = im.InhabitantState.EATS
                end = currentTimeslot._replace(Hour = 13, Minute = 15).to_minutes()
                self.stateEnd = im.stateEnd(end, end)

        elif(currentTimeslot.Hour >= 13 and currentTimeslot.Hour <= 14):
            # Relaxes after lunch until 15:10
            self.state = im.InhabitantState.RELAXES
            end = currentTimeslot._replace(Hour = 15, Minute = 10).to_minutes()
            self.stateEnd = im.stateEnd(end, end)

        elif(currentTimeslot.Hour >= 15 and currentTimeslot.Hour <= 16):
            # Goes for a walk and arrives at 17:30
            self.state = im.InhabitantState.PREPARES_TO_LEAVE
            if(currentState == im.InhabitantState.PREPARES_TO_LEAVE):
                # Leaves for a walk
                self.state = im.InhabitantState.LEFT
                end = currentTimeslot._replace(Hour = 17, Minute = 30).to_minutes()
                self.stateEnd = im.stateEnd(end, end)
            elif(currentState == im.InhabitantState.LEFT):
                # Arrives from a walk
                self.state = im.InhabitantState.ARRIVES

        elif(currentTimeslot.Hour >= 17 and currentTimeslot.Hour <= 19):
            # Works at home until 20:10
            self.state = im.InhabitantState.WORKS
            end = currentTimeslot._replace(Hour = 20, Minute = 10).to_minutes()
            self.stateEnd = im.stateEnd(end, end)

        elif(currentTimeslot.Hour == 20):
            # Does hobby until 21:00
            self.state = im.InhabitantState.DOES_HOBBY
            end = currentTimeslot._replace(Hour = 21, Minute = 0).to_minutes()
            self.stateEnd = im.stateEnd(end, end)

        elif(currentTimeslot.Hour >= 21 and currentTimeslot.Hour <= 22):
            # Prepares food, eats and relaxes until 23:15
            self.state = im.InhabitantState.PREPARES_FOOD
            if(currentState == im.InhabitantState.PREPARES_FOOD):
                # Eats
                self.state = im.InhabitantState.EATS
            elif(currentState == im.InhabitantState.EATS):
                # Relaxes after dinner
                self.state = im.InhabitantState.RELAXES
                end = currentTimeslot._replace(Hour = 23, Minute = 15).to_minutes()
                self.stateEnd = im.stateEnd(end, end)
        
        elif(currentTimeslot.Hour == 23):
            # Sleeps until 5:00 (will be prolonged based on the next day being weekend or not)
            self.state = im.InhabitantState.SLEEPS
            end = currentTimeslot._replace(Hour = 5, Minute = 0).to_minutes() + 24*60
            self.stateEnd = im.stateEnd(end, None)

        # Current state
        if(currentState != self.state):
            print(f'{self.name} | Weekend: {self.env.timeslot} - {self.state}')
