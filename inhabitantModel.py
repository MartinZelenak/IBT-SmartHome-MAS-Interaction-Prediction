import simpy
import enum
from typing import Optional, Generator, NamedTuple, Dict, Callable

from Simulation import homeModel as hm
from Simulation.environment import Environment, TimeoutRequest
from Simulation.utils import truncnorm, truncexp


class InhabitantState(enum.Enum):
    SLEEPS = "Sleeps"
    WAKES_UP = "Wakes up"
    PREPARES_TO_LEAVE = "Prepares to leave"
    LEFT = "Left"
    ARRIVES = "Arrives"
    RELAXES = "Relaxes"
    READS = "Reads"
    WORKS = "Works"   # Work at home
    DOES_HOBBY = "Does hobby" # Each inhabitant has its own hobby
    PREPARES_FOOD = "Prepares food"
    EATS = "Eats"
    TAKES_SHOWER = "Takes shower"
    GOES_TO_SLEEP = "Goes to sleep"
    UNKNOWN = "UNKNOWN"

class stateEnd(NamedTuple):
    min: float | None
    max: float | None

    def average(self) -> float:
        if self.min == None or self.max == None:
            return self.min or self.max or 0
        return (self.min + self.max) / 2

class Inhabitant:
    '''Inhabitant of the house. Has its own schedule and behavior.

    ..._state() methods correspond to a particular state (actions/intercations).
                Time spent in a state is elongated until stateEnd.min if set.
                State yields after stateEnd.min are ignored.
                Raises ValueError when stateEnd.max time is exceeded, if stateEnd.max is set.
    ..._behavior() methods correspond to transitions between states.
    current_state_actions() is called in ..._behavior() methods to execute the current state.

    Inherit this class to create a custom inhabitant.
    '''
    def __init__(self, 
                 env: Environment, 
                 name: str,
                 initial_state: InhabitantState = InhabitantState.UNKNOWN) -> None:
        self.env: Environment = env
        self.state: InhabitantState = initial_state
        self.name: str = name
        self.location: hm.Room | None = None
        self.stateMethodMap: Dict[InhabitantState, Callable[[], Generator[simpy.Event, None, None]]] = {
            InhabitantState.SLEEPS: self.sleeps_state,
            InhabitantState.WAKES_UP: self.wakes_up_state,
            InhabitantState.PREPARES_TO_LEAVE: self.prepares_to_leave_state,
            InhabitantState.LEFT: self.left_state,
            InhabitantState.ARRIVES: self.arrives_state,
            InhabitantState.RELAXES: self.relaxes_state,
            InhabitantState.READS: self.reads_state,
            InhabitantState.WORKS: self.works_state,
            InhabitantState.DOES_HOBBY: self.does_hobby_state,
            InhabitantState.PREPARES_FOOD: self.prepares_food_state,
            InhabitantState.EATS: self.eats_state,
            InhabitantState.TAKES_SHOWER: self.takes_shower_state,
            InhabitantState.GOES_TO_SLEEP: self.goes_to_sleep_state,
            InhabitantState.UNKNOWN: self.unknown_state
        }
        self.stateEnd: stateEnd = stateEnd(None, None)
    
    def go_to_room(self, room_name: str) -> bool:
        '''Returns True if the inhabitant moved to the room, False otherwise.'''
        if not self.location or self.location.name != room_name:
            self.location = self.env.home.go_to_room(room_name, self.name)
            return True
        return False


    def sleeps_state(self) -> Generator[simpy.Event, None, None]:
        yield self.env.timeout(1)

    def wakes_up_state(self) -> Generator[simpy.Event, None, None]:
        yield self.env.timeout(1)

    def prepares_to_leave_state(self) -> Generator[simpy.Event, None, None]:
        yield self.env.timeout(1)

    def left_state(self) -> Generator[simpy.Event, None, None]:
        yield self.env.timeout(1)

    def arrives_state(self) -> Generator[simpy.Event, None, None]:
        yield self.env.timeout(1)

    def relaxes_state(self) -> Generator[simpy.Event, None, None]:
        yield self.env.timeout(1)

    def reads_state(self) -> Generator[simpy.Event, None, None]:
        yield self.env.timeout(1)

    def works_state(self) -> Generator[simpy.Event, None, None]:
        yield self.env.timeout(1)

    def does_hobby_state(self) -> Generator[simpy.Event, None, None]:
        yield self.env.timeout(1)

    def prepares_food_state(self) -> Generator[simpy.Event, None, None]:
        yield self.env.timeout(1)

    def eats_state(self) -> Generator[simpy.Event, None, None]:
        yield self.env.timeout(1)

    def takes_shower_state(self) -> Generator[simpy.Event, None, None]:
        yield self.env.timeout(1)

    def goes_to_sleep_state(self) -> Generator[simpy.Event, None, None]:
        yield self.env.timeout(1)

    def unknown_state(self):
        raise ValueError(f'Tried to execute unknown state {self.state}!')


    def current_state_actions(self) -> Generator[simpy.Event, None, None]:
        '''Executes the current state actions and yields events until the state ends.
        stateEnd.min and stateEnd.max can be used to determine the state end.
        If state ends before stateEnd.min, the state is prolonged until truncexp(stateEnd.min - stateEnd.max) or stateEnd.min if stateEnd.max is None.
        If state ends after stateEnd.max, the last timout yield, that goes beyond stateEnd.max, is ignored (state is shortened).
        WARNING: After ignoring a timeout yield, the ..._state() method will continue until the next yield/return.
        '''

        # Current state event generator
        stateYield = self.stateMethodMap.get(self.state, self.unknown_state)()

        if self.stateEnd.max != None:
            # States are cut short if they are too long
            while self.env.now < self.stateEnd.max:
                try:
                    event = next(stateYield)
                    
                    # Handle TimeoutRequests
                    if isinstance(event, TimeoutRequest):
                        # Ignore TimeoutRequest if over max time
                        if event._delay + self.env.now > self.stateEnd.max:
                            stateYield.close()
                            raise StopIteration
                        else:
                            event.confirm()
                    # Ignore Timeout if over max time
                    elif isinstance(event, simpy.Timeout):
                        if event._delay + self.env.now > self.stateEnd.max:
                            # WARNING: The ..._state() method will continue until the next yield/return
                            stateYield.close()
                            raise StopIteration
                    yield event
                except StopIteration:
                    # States are extended if they are too short
                    # (if stateEnd.min is set)
                    if self.stateEnd.min != None and self.env.now < self.stateEnd.min:
                        minMaxDiff = self.stateEnd.max - self.stateEnd.min
                        end = truncexp(minMaxDiff / 2, minMaxDiff) + self.stateEnd.min
                        yield self.env.timeout(end - self.env.now)
                    break

            if self.env.now > self.stateEnd.max:
                raise ValueError(f'Current state {self.state} event took too long!\nState ended: {self.env.now}\nEnd interval: {self.stateEnd}')
            
        elif self.stateEnd.min != None: # and self.stateEnd.max == None
            while True:
                try:
                    event = next(stateYield)

                    # Confirm any TimeoutRequest
                    if isinstance(event, TimeoutRequest):
                        event.confirm()
                    
                    yield event
                except StopIteration:
                    # States are extended if they are too short
                    if self.stateEnd.min > self.env.now:
                        yield self.env.timeout(self.stateEnd.min - self.env.now)
                    break
        else:
            try:
                while True:
                    event = next(stateYield)
                    # Confirm any TimeoutRequest
                    if isinstance(event, TimeoutRequest):
                        event.confirm()
                    yield event
            except StopIteration:
                pass
            # yield from stateYield

    def workday_behavior(self) -> Generator[simpy.Event, None, None] | None:
        yield self.env.timeout(1)

    def weekend_behavior(self) -> Generator[simpy.Event, None, None] | None:
        yield self.env.timeout(1)


    def behaviour(self):
        weekendBehavior = False
        workdayBehavior = False
        while True:
            currentState = self.state
            if self.env.is_weekend():
                eventGenerator = self.weekend_behavior()
                weekendBehavior = True
            else:
                eventGenerator = self.workday_behavior()
                workdayBehavior = True
            
            if eventGenerator:
                yield from eventGenerator

            # # Current state actions
            # # (if state changed or if behavior changed)
            # if currentState != self.state:
            #     yield from self.current_state_actions()
            # elif weekendBehavior and workdayBehavior:
            #     yield from self.current_state_actions()
            #     weekendBehavior, workdayBehavior = False, False
            # else:
            #     # Await next state change
            #     yield self.env.timeout(1)

            # Current state actions
            yield from self.current_state_actions()