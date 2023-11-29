import simpy
import random
from typing import Optional, Generator, NamedTuple
import inhabitantModel as im
import homeModel as hm
from environment import TimeSlotEnvironment, TimeSlot
from utils import truncnorm, truncexp
import random

ROOMS = ['livingroom', 'kitchen', 'bathroom', 'bedroom', 'office', 'hallway', 'outside']

class ScenarioInhabitant(im.Inhabitant):
    def __init__(self, env: TimeSlotEnvironment, name: str, home: Optional[hm.Home] = None) -> None:
        super().__init__(env, home)
        self.name = name
        # TODO: self.stateofmind = ..

    def sleeps_state(self) -> Generator[simpy.Event, None, None]:
        '''Sleeps state must be prolonged'''
        yield self.env.timeoutRequest(0)

    def wakes_up_state(self) -> Generator[simpy.Event, None, None]:
        if(self.env.is_weekend()):
            # Put on home clothes
            yield self.env.timeoutRequest(2 + truncexp(1.5, None, 3)) # 2-5 minutes

        # Go to the bathroom
        self.location = self.home.go_to_room('bathroom')
        yield self.env.timeoutRequest(0.5 + truncexp(0.25, None, 0.5)) # 0.25-0.5 minutes
        
        # Brush teeth
        yield self.env.timeoutRequest(1 + truncexp(1, None, 2)) # 1-3 minutes

        # Make coffee ???

        
    def prepares_to_leave_state(self) -> Generator[simpy.Event, None, None]:
        # Put on clothes and grab stuff
        yield self.env.timeoutRequest(3 + truncexp(2, None, 4)) # 3-7 minutes

        # Go to the door
        self.location = self.home.go_to_room('hallway')
        yield self.env.timeoutRequest(2 + truncexp(0.5, None, 1)) # 2-3 minutes

        # Put on shoes
        yield self.env.timeoutRequest(1 + truncexp(0.5, None, 1)) # 1-2 minutes

    def left_state(self) -> Generator[simpy.Event, None, None]:
        '''Left state must be prolonged'''
        self.location = self.home.go_to_room('outside')
        yield self.env.timeoutRequest(0)

    def arrives_state(self) -> Generator[simpy.Event, None, None]:
        # Enter the hallway
        self.location = self.home.go_to_room('hallway')
        
        # Put off shoes
        yield self.env.timeoutRequest(0.5 + truncexp(0.25, None, 0.5)) # 0.5-1 minutes

    def relaxes_state(self) -> Generator[simpy.Event, None, None]:
        '''Relaxes state must be cut short'''
        # Go to the livingroom
        self.location = self.home.go_to_room('livingroom')
        yield self.env.timeoutRequest(0.5 + truncexp(0.25, None, 0.5)) # 0.5-1 minutes

        # Sit on the couch
        yield self.env.timeoutRequest(0.5 + truncexp(0.25, None, 0.5)) # 0.5-1 minutes
        
        # Turn on the TV
        # TODO: self.location.devices[0].turn_on()

        # Watch TV
        while True:
            # Will be cut short by stateEnd.max
            # TODO: truncexp(15, None, 30) High enough variance? How to determine?
            yield self.env.timeoutRequest(truncexp(15, None, 30)) # 0-30 minutes

    def reads_state(self) -> Generator[simpy.Event, None, None]:
        # Go to the livingroom
        self.location = self.home.go_to_room('livingroom')
        yield self.env.timeoutRequest(0.5 + truncexp(0.25, None, 0.5)) # 0.5-1 minutes

        # Sit on the couch
        yield self.env.timeoutRequest(0.5 + truncexp(0.25, None, 0.5)) # 0.5-1 minutes

        # Read a book
        while True:
            # Will be cut short by stateEnd.max
            yield self.env.timeoutRequest(truncexp(10, None, 25)) # 0-25 minutes

    def does_hobby_state(self) -> Generator[simpy.Event, None, None]:
        '''Scrolls through his/her phone in the bedroom'''
        # Go to the bedroom
        self.location = self.home.go_to_room('bedroom')

        # Lay on the bed
        yield self.env.timeoutRequest(0.5 + truncexp(0.20, None, 0.4)) # 0.5-0.9 minutes

        # Scroll through phone
        while True:
            # Will be cut short by stateEnd.max
            yield self.env.timeoutRequest(truncexp(20, None, 40)) # 0-40 minutes

    def works_state(self) -> Generator[simpy.Event, None, None]:
        # Go to the office
        self.location = self.home.go_to_room('office')
        yield self.env.timeoutRequest(0.5 + truncexp(0.25, None, 0.5)) # 0.5-1 minutes

        # Sit on the chair
        yield self.env.timeoutRequest(0.2 + truncexp(0.25, None, 0.5)) # 0.2-0.5 minutes

        # Work
        while True:
            # Will be cut short by stateEnd.max
            yield self.env.timeoutRequest(truncexp(30, None, 60)) # 0-60 minutes

    def prepares_food_state(self) -> Generator[simpy.Event, None, None]:
        # Go to the kitchen
        self.location = self.home.go_to_room('kitchen')
        yield self.env.timeoutRequest(0.5 + truncexp(0.25, None, 0.5)) # 0.5-1 minutes

        # Prepare food
        yield self.env.timeoutRequest(20 + truncexp(12.5, None, 25)) # 20-45 minutes

    def eats_state(self) -> Generator[simpy.Event, None, None]:
        # Go to the livingroom
        self.location = self.home.go_to_room('livingroom')
        yield self.env.timeoutRequest(0.5 + truncexp(0.25, None, 0.5)) # 0.5-1 minutes

        # Sit on a chair
        yield self.env.timeoutRequest(0.3 + truncexp(0.30, None, 0.6)) # 0.3-1.1 minutes

        # Eat
        yield self.env.timeoutRequest(15 + truncexp(7.5, None, 15)) # 15-30 minutes


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
            end = currentTimeslot._replace(Hour = 6, Minute = truncexp(7.5, None, 15)).to_minutes()
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
            # Leaves until 16:00 - 16:45
            self.state = im.InhabitantState.LEFT
            endMin = currentTimeslot._replace(Hour = 16, Minute = 0).to_minutes()
            endMax = currentTimeslot._replace(Hour = 16, Minute = 45).to_minutes()
            self.stateEnd = im.stateEnd(endMin, endMax)

        elif(currentState == im.InhabitantState.LEFT):
            self.state = im.InhabitantState.ARRIVES
            
        elif((currentTimeslot.Hour >= 16 and currentTimeslot.Hour <= 20) or currentState == im.InhabitantState.ARRIVES):
            # Choose a random state with some distribution (different than current state)
            while True:
                choice = random.random()
                if choice < 0.4:
                # 40%
                    # TODO Remotely turn on the livingroom lights (knows he is going there)
                    self.state = im.InhabitantState.RELAXES
                elif choice < 0.7:
                # 30%
                    self.state = im.InhabitantState.READS
                elif choice < 0.9:
                # 20%
                    self.state = im.InhabitantState.DOES_HOBBY
                else:
                # 10%
                    self.state = im.InhabitantState.WORKS

                if(self.state != currentState):
                    break

            endMax = currentTimeslot._replace(Hour = 21, Minute = 30).to_minutes()
            if endMax - self.env.now > 65:
                nextStateEnd = self.env.now + truncexp((endMax - self.env.now) / 2, 60, endMax - self.env.now)
            else:
                nextStateEnd = self.env.now + truncexp((endMax - self.env.now) / 2, None, 60)
            self.stateEnd = im.stateEnd(None, nextStateEnd)

        elif(currentTimeslot.Hour == 21):
            self.state = im.InhabitantState.PREPARES_FOOD
            endMax = currentTimeslot._replace(Hour = 22, Minute = 15).to_minutes()
            self.stateEnd = im.stateEnd(None, endMax)
        
        elif(currentTimeslot.Hour == 22 and currentTimeslot.Minute < 25):
            # Eats until 22:30 - 23:00
            self.state = im.InhabitantState.EATS
            endMin = currentTimeslot._replace(Hour = 22, Minute = 30).to_minutes()
            endMax = currentTimeslot._replace(Hour = 23, Minute = 0).to_minutes()
            self.stateEnd = im.stateEnd(endMin, endMax)
        
        elif((currentTimeslot.Hour == 22 and currentTimeslot.Minute >= 25) or currentTimeslot.Hour >= 23):
            # Sleeps until 6:00-6:15
            self.state = im.InhabitantState.SLEEPS
            end = currentTimeslot._replace(Hour = 6, Minute = truncexp(7.5, None, 15)).to_minutes() + 24*60
            self.stateEnd = im.stateEnd(end, None)

        # Current state
        if(currentState != self.state):
            print(f'{self.name} | Workday: {self.env.timeslot} - {self.state}')
            


    def weekend_behavior(self) -> Generator[simpy.Event, None, None] | None:
        # print(f'{self.name} | Weekend: {self.env.timeslot} - {self.state}')
        self.state = im.InhabitantState.SLEEPS
        self.stateEnd = im.stateEnd(self.env.now + 24*60, None)


def setup_home(env: TimeSlotEnvironment) -> hm.Home:
    home = hm.Home(env)

    for roomName in ROOMS:
        room = hm.Room(env, roomName)
        room.add_device(hm.SmartDevice(env, f'{roomName}_light'))
        home.add_room(room)
    
    return home


if __name__ == '__main__':
    env = TimeSlotEnvironment()
    home = setup_home(env)
    inhabitant = ScenarioInhabitant(env, home)
    
    env.process(inhabitant.behaviour())

    env.run(60*24) # Run for a workday
    # env.run(60*24*5) # Run for 5 workdays

    print('Finish time: ' + str(env.timeslot))