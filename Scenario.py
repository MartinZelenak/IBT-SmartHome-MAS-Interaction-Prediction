import simpy
import random
from typing import Optional, Generator, NamedTuple
import inhabitantModel as im
import homeModel as hm
import deviceModels as dm
from environment import Environment, TimeSlot
from utils import truncnorm, truncexp

ROOMS = ['livingroom', 'kitchen', 'bathroom', 'bedroom', 'office', 'hallway', 'outside']

SIM_START = TimeSlot(Minute=0, Hour=0, Day=1, Month=1, Year=0).to_minutes()
SIM_END   = TimeSlot(Minute=0, Hour=0, Day=8, Month=1, Year=0).to_minutes()

NUM_OF_INHABITANTS = 1

class ScenarioInhabitant(im.Inhabitant):
    def __init__(self, env: Environment, name: str, home: Optional[hm.Home] = None) -> None:
        super().__init__(env, name)
        # TODO: self.stateofmind = ..

    def sleeps_state(self) -> Generator[simpy.Event, None, None]:
        '''Sleeps state must be prolonged'''
        # Go to the bedroom
        if(self.go_to_room('bedroom')):
            yield self.env.timeoutRequest(0.5 + truncexp(0.25, None, 0.5)) # 0.5-1 minutes

        yield self.env.timeoutRequest(0)

    def wakes_up_state(self) -> Generator[simpy.Event, None, None]:
        if(self.env.is_weekend()):
            # Doesn't get up immediately
            yield self.env.timeoutRequest(3 + truncexp(7.5, None, 15)) # 3-15 minutes

            ## Turns the lights on
            self.env.home.get_device_op('bedroom', 'bedroom_light', 'turn_on')(self.name)

            # Put on home clothes
            yield self.env.timeoutRequest(2 + truncexp(1.5, None, 3)) # 2-5 minutes

            ## Turns the lights off
            self.env.home.get_device_op('bedroom', 'bedroom_light', 'turn_off')(self.name)

        # Go to the bathroom
        if(self.go_to_room('bathroom')):
            yield self.env.timeoutRequest(0.5 + truncexp(0.25, None, 0.5)) # 0.25-0.5 minutes
        
        # Brush teeth
        yield self.env.timeoutRequest(1 + truncexp(1, None, 2)) # 1-3 minutes

        # Go to the kitchen
        self.go_to_room('kitchen')
        yield self.env.timeoutRequest(0.5 + truncexp(0.25, None, 0.5)) # 0.5-1 minutes

        # Make coffee
        yield self.env.timeoutRequest(2 + truncexp(1.5, None, 3)) # 2-5 minutes

        
    def prepares_to_leave_state(self) -> Generator[simpy.Event, None, None]:
        # Go to the bedroom
        if(self.go_to_room('bedroom')):
            yield self.env.timeoutRequest(0.5 + truncexp(0.25, None, 0.5)) # 0.5-1 minutes

        ## Turns the lights on
        lightIsOn = False
        if(self.env.timeslot.Hour < 8 or self.env.timeslot.Hour >= 21):
            self.env.home.get_device_op('bedroom', 'bedroom_light', 'turn_on')(self.name)
            lightIsOn = True

        # Put on clothes and grab stuff
        yield self.env.timeoutRequest(3 + truncexp(2, None, 4)) # 3-7 minutes

        ## Turns the lights off
        if(lightIsOn):
            self.env.home.get_device_op('bedroom', 'bedroom_light', 'turn_off')(self.name)

        # Go to the door
        self.go_to_room('hallway')
        yield self.env.timeoutRequest(2 + truncexp(0.5, None, 1)) # 2-3 minutes

        # Put on shoes
        yield self.env.timeoutRequest(1 + truncexp(0.5, None, 1)) # 1-2 minutes

    def left_state(self) -> Generator[simpy.Event, None, None]:
        '''Left state must be prolonged'''
        self.go_to_room('outside')
        yield self.env.timeoutRequest(0)

    def arrives_state(self) -> Generator[simpy.Event, None, None]:
        # Enter the hallway
        self.go_to_room('hallway')
        
        # Put off shoes
        yield self.env.timeoutRequest(0.5 + truncexp(0.25, None, 0.5)) # 0.5-1 minutes

        # Go to the bedroom
        self.go_to_room('bedroom')

        # Put off clothes and put on home clothes
        yield self.env.timeoutRequest(3 + truncexp(2, None, 4)) # 3-7 minutes

    def relaxes_state(self) -> Generator[simpy.Event, None, None]:
        '''Relaxes state must be cut short'''
        # Go to the livingroom
        if(self.go_to_room('livingroom')):
            yield self.env.timeoutRequest(0.5 + truncexp(0.25, None, 0.5)) # 0.5-1 minutes

        # Sit on the couch
        yield self.env.timeoutRequest(0.5 + truncexp(0.25, None, 0.5)) # 0.5-1 minutes
        
        # Turn on the TV
        # TODO: self.location.devices[0].turn_on()

        # Watch TV
        while True:
            # Will be cut short by stateEnd.max
            yield self.env.timeoutRequest(truncexp(15, None, 30)) # 0-30 minutes

    def reads_state(self) -> Generator[simpy.Event, None, None]:
        '''Reads state must be cut short'''
        # Go to the livingroom
        if(self.go_to_room('livingroom')):
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
        if(self.go_to_room('bedroom')):
            yield self.env.timeoutRequest(0.5 + truncexp(0.25, None, 0.5)) # 0.5-1 minutes

        # Lay on the bed
        yield self.env.timeoutRequest(0.5 + truncexp(0.20, None, 0.4)) # 0.5-0.9 minutes

        # Scroll through phone
        while True:
            # Will be cut short by stateEnd.max
            yield self.env.timeoutRequest(truncexp(20, None, 40)) # 0-40 minutes

    def works_state(self) -> Generator[simpy.Event, None, None]:
        # Go to the office
        if(self.go_to_room('office')):
            yield self.env.timeoutRequest(0.5 + truncexp(0.25, None, 0.5)) # 0.5-1 minutes

        # Sit on the chair
        yield self.env.timeoutRequest(0.2 + truncexp(0.25, None, 0.5)) # 0.2-0.5 minutes

        # Work
        while True:
            # Will be cut short by stateEnd.max
            yield self.env.timeoutRequest(truncexp(30, None, 60)) # 0-60 minutes

    def prepares_food_state(self) -> Generator[simpy.Event, None, None]:
        # Go to the kitchen
        if(self.go_to_room('kitchen')):
            yield self.env.timeoutRequest(0.5 + truncexp(0.25, None, 0.5)) # 0.5-1 minutes

        # Prepare food
        yield self.env.timeoutRequest(20 + truncexp(12.5, None, 25)) # 20-45 minutes

    def eats_state(self) -> Generator[simpy.Event, None, None]:
        # Go to the livingroom
        if(self.go_to_room('livingroom')):
            yield self.env.timeoutRequest(0.5 + truncexp(0.25, None, 0.5)) # 0.5-1 minutes

        # Sit on a chair
        yield self.env.timeoutRequest(0.3 + truncexp(0.30, None, 0.6)) # 0.3-1.1 minutes

        # Eat
        yield self.env.timeoutRequest(15 + truncexp(7.5, None, 15)) # 15-30 minutes


    def workday_behavior(self) -> Generator[simpy.Event, None, None] | None:
        # Next state logic
        currentTimeslot = self.env.timeslot
        currentState = self.state
        self.state = im.InhabitantState.UNKNOWN # Default state
        self.stateEnd = im.stateEnd(None, None) # Reset state end
        if(currentTimeslot.Hour < 6):
            # Sleeps until 6:00-6:15
            self.state = im.InhabitantState.SLEEPS
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
                # TODO: Last activity too short?
                nextStateEnd = self.env.now + truncexp((endMax - self.env.now) / 2, None, 60)
            self.stateEnd = im.stateEnd(None, nextStateEnd)

        elif(currentTimeslot.Hour == 21):
            self.state = im.InhabitantState.PREPARES_FOOD
            endMin = currentTimeslot._replace(Hour = 22, Minute = 15).to_minutes()
            self.stateEnd = im.stateEnd(endMin, None)
        
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
        # Next state logic
        currentTimeslot = self.env.timeslot
        currentState = self.state
        self.state = im.InhabitantState.UNKNOWN # Default state
        self.stateEnd = im.stateEnd(None, None) # Reset state end
        if(currentTimeslot.Hour < 8):
            # Sleeps until 8:00-9:30
            self.state = im.InhabitantState.SLEEPS
            end = currentTimeslot._replace(Hour = 8, Minute = truncexp(45, None, 90)).to_minutes()
            self.stateEnd = im.stateEnd(end, None)

        elif(currentTimeslot.Hour >= 8 and currentTimeslot.Hour <= 9):
            # Wakes up after Sleeping
            if(currentState == im.InhabitantState.SLEEPS):
                self.state = im.InhabitantState.WAKES_UP
            else:
                # Choose a random state with some distribution
                choice = random.random()
                if choice < 0.45:
                # 45%
                    self.state = im.InhabitantState.DOES_HOBBY
                elif choice < 0.75:
                # 35%
                    self.state = im.InhabitantState.READS
                else:
                # 25%
                    self.state = im.InhabitantState.WORKS

                endMax = currentTimeslot._replace(Hour = 12, Minute = 00).to_minutes()
                self.stateEnd = im.stateEnd(None, endMax)

        elif(currentTimeslot.Hour >= 11 and currentTimeslot.Hour <= 12):
            # Prepares food until 13:00-13:30
            self.state = im.InhabitantState.PREPARES_FOOD
            endMin = currentTimeslot._replace(Hour = 13, Minute = 0).to_minutes()
            endMax = currentTimeslot._replace(Hour = 13, Minute = 30).to_minutes()
            self.stateEnd = im.stateEnd(endMin, endMax)

        elif(currentTimeslot.Hour >= 13 and currentTimeslot.Hour <= 14):
            if(currentState == im.InhabitantState.PREPARES_FOOD):
                # Eats
                self.state = im.InhabitantState.EATS
            elif(currentState == im.InhabitantState.EATS):
                # Relaxes after lunch
                self.state = im.InhabitantState.RELAXES
                endMin = currentTimeslot._replace(Hour = 15, Minute = 0).to_minutes()
                endMax = currentTimeslot._replace(Hour = 15, Minute = 20).to_minutes()
                self.stateEnd = im.stateEnd(endMin, endMax)

        elif(currentTimeslot.Hour >= 15 and currentTimeslot.Hour <= 18):
            if(currentState == im.InhabitantState.PREPARES_TO_LEAVE):
                # Leaves for a walk
                self.state = im.InhabitantState.LEFT
            elif(currentState == im.InhabitantState.LEFT):
                # Arrives from a walk
                self.state = im.InhabitantState.ARRIVES
            else:
                # Choose a random state with some distribution (different than current state)
                while True:
                    choice = random.random()
                    if choice < 0.4:
                    # 40%
                        self.state = im.InhabitantState.DOES_HOBBY
                    elif choice < 0.65:
                    # 25%
                        self.state = im.InhabitantState.WORKS
                    elif choice < 0.85:
                    # 20%
                        self.state = im.InhabitantState.READS
                    else:
                    # 15%
                        # Goes for a walk
                        self.state = im.InhabitantState.PREPARES_TO_LEAVE

                    if(self.state != currentState):
                        break
            
            # Does 2-3 activities
            if(self.state != im.InhabitantState.PREPARES_TO_LEAVE 
               and self.state != im.InhabitantState.ARRIVES):
                if(currentTimeslot.Hour == 15):
                    endMin = currentTimeslot._replace(Hour = 16, Minute = 20).to_minutes()
                    endMax = currentTimeslot._replace(Hour = 17, Minute = 15).to_minutes()
                elif(currentTimeslot.Hour == 16):
                    endMin = currentTimeslot._replace(Hour = 17, Minute = 55).to_minutes()
                    endMax = currentTimeslot._replace(Hour = 19, Minute = 5).to_minutes()
                else:
                    endMin = currentTimeslot._replace(Hour = 19, Minute = 30).to_minutes()
                    endMax = currentTimeslot._replace(Hour = 19, Minute = 59).to_minutes()
                self.stateEnd = im.stateEnd(endMin, endMax)

        elif(currentTimeslot.Hour == 19):
            # Works or Does hobby after previous activities
            choice = random.random()
            if choice < 0.7:
            # 70%
                self.state = im.InhabitantState.WORKS
            else:
            # 30%
                self.state = im.InhabitantState.DOES_HOBBY

            endMin = currentTimeslot._replace(Hour = 20, Minute = 5).to_minutes()
            endMax = currentTimeslot._replace(Hour = 21, Minute = 20).to_minutes()
            self.stateEnd = im.stateEnd(endMin, endMax)

        elif(currentTimeslot.Hour == 20): # May be skipped if previous activity was long enough
            # Reads
            self.state = im.InhabitantState.READS
            endMax = currentTimeslot._replace(Hour = 21, Minute = 30).to_minutes()
            self.stateEnd = im.stateEnd(None, endMax)

            ## Remotly turns on the livingroom light (knows he is going there)¨
            self.env.home.get_device_op('livingroom', 'livingroom_light', 'turn_on')(self.name)

        elif(currentTimeslot.Hour >= 21 and currentTimeslot.Hour <= 22):
            # Prepares food
            self.state = im.InhabitantState.PREPARES_FOOD

            if(currentState == im.InhabitantState.PREPARES_FOOD):
                # Eats
                self.state = im.InhabitantState.EATS
            elif(currentState == im.InhabitantState.EATS):
                # Relaxes after dinner
                self.state = im.InhabitantState.RELAXES
                endMin = currentTimeslot._replace(Hour = 23, Minute = 0).to_minutes()
                endMax = currentTimeslot._replace(Hour = 23, Minute = 30).to_minutes()
                self.stateEnd = im.stateEnd(endMin, endMax)
        
        elif(currentTimeslot.Hour == 23):
            # Sleeps until 5:00 (will be prolonged based on the next day being weekend or not)
            self.state = im.InhabitantState.SLEEPS
            end = currentTimeslot._replace(Hour = 5, Minute = 0).to_minutes() + 24*60
            self.stateEnd = im.stateEnd(end, None)

        # Current state
        if(currentState != self.state):
            print(f'{self.name} | Weekend: {self.env.timeslot} - {self.state}')


def setup_home(home: hm.Home):
    for roomName in ROOMS:
        room = hm.Room(env, roomName)
        room.add_device(dm.SmartLight(env, f'{roomName}_light'))
        home.add_room(room)

def day_divider(env: Environment) -> Generator[simpy.Event, None, None]:
    '''Prints the current day every 24 hours'''
    while True:
        timeslot = env.timeslot
        print(f'DAY {timeslot.Day}.{timeslot.Month}.{timeslot.Year} ----------------------------------------------')
        yield env.timeout(60*24)


if __name__ == '__main__':
    # Environment
    env = Environment(SIM_START)
    setup_home(env.home)

    # Day dividning prints
    env.process(day_divider(env))
    
    # Inhabitants
    for i in range(1, NUM_OF_INHABITANTS + 1):
        inhabitant = ScenarioInhabitant(env, str(i))
        inhabitant.location = env.home.rooms['bedroom'] # Start in the bedroom
        env.process(inhabitant.behaviour())
    
    env.run(SIM_END)

    print('Finish time: ' + str(env.timeslot))