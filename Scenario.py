import simpy
import random
from typing import Optional, Generator, NamedTuple
import inhabitantModel as im
import homeModel as hm
from environment import TimeSlotEnvironment, TimeSlot
from utils import truncnorm, truncexp

ROOMS = ['livingroom', 'kitchen', 'bathroom', 'bedroom', 'hallway', 'outside']

class ScenarioInhabitant(im.Inhabitant):
    def __init__(self, env: TimeSlotEnvironment, home: Optional[hm.Home] = None) -> None:
        super().__init__(env, home)
        # TODO: self.stateofmind = ..

    def sleeps_state(self) -> Generator[simpy.Event, None, None]:
        '''Sleeps state. Will be prolonged until stateEnd.min'''
        print(f'- Sleeps: {self.env.timeslot}')
        yield self.env.timeout(0) # Sleep until stateEnd.min
        
    def prepares_to_leave_state(self) -> Generator[simpy.Event, None, None]:
        print(f'- Prepares to leave: {self.env.timeslot}')
        
        # Put on clothes
        yield self.env.timeout(3 + truncexp(2, None, 4)) # 3-7 minutes

        # Go to the door
        # TODO: Go to the door
        yield self.env.timeout(2 + truncexp(0.5, None, 1)) # 2-3 minutes


    def workday_behavior(self) -> Generator[simpy.Event, None, None] | None:
        self.stateEnd = im.stateEnd(None, None) # Reset state end
        
        # Next state logic
        currentTimeslot = self.env.timeslot
        currentState = self.state
        self.state = im.InhabitantState.UNKNOWN # Default state
        if(currentTimeslot.Hour < 6):
            # Sleeps until 6:00-6:15
            self.state = im.InhabitantState.SLEEPS
            # TODO: truncexp() vs truncnorm()?
            end = currentTimeslot._replace(Hour = 6, Minute = truncexp(7.5, 0, 15)).to_minutes()
            self.stateEnd = im.stateEnd(end, None)

        elif(currentTimeslot.Hour == 6):
            # Wakes up after Sleeping
            if(currentState == im.InhabitantState.SLEEPS):
                self.state = im.InhabitantState.WAKES_UP
            # Prepares to leave until 7:00 - 7:15
            elif(currentState == im.InhabitantState.WAKES_UP):
                self.state = im.InhabitantState.PREPARES_TO_LEAVE
                endMin = currentTimeslot._replace(Hour = 7, Minute = 0).to_minutes()
                endMax = currentTimeslot._replace(Hour = 7, Minute = 15).to_minutes()
                self.stateEnd = im.stateEnd(endMin, endMax)
            else:
                self.state = im.InhabitantState.LEFT
        
        elif(currentTimeslot.Hour >= 7 and currentTimeslot.Hour <= 15):
            self.state = im.InhabitantState.LEFT
        elif(currentState == im.InhabitantState.LEFT):
            self.state = im.InhabitantState.ARRIVES
        elif(currentState == im.InhabitantState.ARRIVES):
            # TODO Remotly turn on the livingroom lights (knows he is going there)
            self.state = im.InhabitantState.RELAXES # Will go to the livingroom
        elif(currentState == im.InhabitantState.RELAXES):
            self.state = im.InhabitantState.SLEEPS
            end = (TimeSlot.from_minutes(self.env.now + 24*60))._replace(Hour = 6, Minute = truncexp(7.5, 0, 15)).to_minutes()
            self.stateEnd = im.stateEnd(end, None)

        # Current state
        if(currentState != self.state):
            print(f'Workday: {self.env.timeslot} - {self.state}')
            


    def weekend_behavior(self) -> Generator[simpy.Event, None, None] | None:
        # print(f'Weekend: {self.env.timeslot} - {self.state}')
        self.state = im.InhabitantState.SLEEPS
        self.stateEnd = im.stateEnd(self.env.now + 24*60, None)


def setup_home(env: TimeSlotEnvironment) -> hm.Home:
    home = hm.Home(env)

    for roomName in ROOMS:
        room = hm.Room(env, roomName)
        room.add_device(hm.SmartDevice(env, f'{roomName}_light'))
        home.add_room(room)
    
    return home


def clock(env: TimeSlotEnvironment) -> Generator[simpy.Event, None, None]:
    while True:
        yield env.timeout(1)
        print(f'Timeslot: {env.timeslot}')


if __name__ == '__main__':
    env = TimeSlotEnvironment()
    home = setup_home(env)
    inhabitant = ScenarioInhabitant(env, home)
    
    env.process(inhabitant.behaviour())
    # env.process(clock(env))

    env.run(60*24) # Run for a day
    # env.run(60*24*5) # Run for 5 workdays
    # env.run(60*24*7) # Run for a week
    # env.run(60*24*7*2) # Run for 2 weeks

    print('Finish time: ' + str(env.timeslot))