import simpy
import random
from typing import Optional, Generator, NamedTuple
import inhabitantModel as im
import homeModel as hm
from environment import TimeSlotEnvironment, TimeSlot
from utils import truncnorm

ROOMS = ['livingroom', 'kitchen', 'bathroom', 'bedroom', 'hallway', 'outside']

class ScenarioInhabitant(im.Inhabitant):
    def __init__(self, env: TimeSlotEnvironment, home: Optional[hm.Home] = None) -> None:
        super().__init__(env, home)
        # TODO: self.stateofmind = ..

    def sleeps_state(self) -> Generator[simpy.Event, None, None]:
        '''Sleeps state. Will be prolonged until stateEnd.min'''
        print(f"Sleeping Zzz... {self.env.timeslot} - {self.state}")
        yield self.env.timeout(0) # Sleep until stateEnd.min
        
    def prepares_to_leave_state(self) -> Generator[simpy.Event, None, None]:
        print(f'Prepares to leave: {self.env.timeslot}')
        
        # Put on clothes
        yield self.env.timeout(truncnorm(5, 2, 3, 7)) # 3-7 minutes

        # Go to the door
        # TODO: Go to the door
        yield self.env.timeout(random.uniform(2, 3))


    def workday_behavior(self) -> Generator[simpy.Event, None, None] | None:
        self.stateEnd = im.stateEnd(None, None) # Reset state end
        
        # Next state logic
        currentState = self.state
        currentTimeslot = self.env.timeslot
        if(currentTimeslot.Hour < 6):
            # Sleeps until 6:00-6:30
            self.state = im.InhabitantState.SLEEPS
            endMin = currentTimeslot._replace(Hour = 6, Minute = 0).to_minutes() # Today at 6:00
            end = truncnorm(endMin + 15, 5, endMin, endMin + 30) # 6:00-6:30
            self.stateEnd = im.stateEnd(end, None)

        elif(currentTimeslot.Hour == 6):
            if(currentTimeslot.Minute < 30 and self.state == im.InhabitantState.SLEEPS):
                self.state = im.InhabitantState.WAKES_UP
            elif(self.state == im.InhabitantState.WAKES_UP):
                self.state = im.InhabitantState.PREPARES_TO_LEAVE
                self.stateEnd = im.stateEnd(self.env.now + 30, self.env.now + 35)  # Prepares to leave for 30-35 minutes
        elif(currentTimeslot.Hour >= 7 and currentTimeslot.Hour <= 15):
            self.state = im.InhabitantState.LEFT
        elif(self.state == im.InhabitantState.LEFT):
            self.state = im.InhabitantState.ARRIVES
        elif(self.state == im.InhabitantState.ARRIVES):
            # Remotly turn on the livingroom lights (knows he is going there)
            # TODO
            self.state = im.InhabitantState.RELAXES # Will go to the livingroom
        elif(self.state == im.InhabitantState.RELAXES):
            self.state = im.InhabitantState.SLEEPS
            endMin = TimeSlot.from_minutes(self.env.now + 24*60)._replace(Hour = 6, Minute = 0).to_minutes()
            self.stateEnd = im.stateEnd(endMin, endMin + 15)  # Sleeps until 6:00-6:15
        else:
            self.state = im.InhabitantState.UNKNOWN

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