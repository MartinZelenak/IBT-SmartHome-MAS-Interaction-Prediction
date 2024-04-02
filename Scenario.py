import simpy
from typing import Generator, List

from . import homeModel as hm
from . import deviceModels as dm
from .environment import Environment, Time
from .stateLogger import StateLogger
from .inhabitants.stochastic import StochasticInhabitant, ROOMS

SIM_START = Time(Minute=0, Hour=0, Day=1, Month=1, Year=1).to_minutes()
SIM_END   = Time(Minute=0, Hour=0, Day=2, Month=1, Year=1).to_minutes()
SIM_ITERATIONS = 1

LOG_TIME_INTERVAL = 5 # minutes # Log every LOG_TIME_INTERVAL minutes

NUM_OF_INHABITANTS = 1

def setup_home(home: hm.Home):
    for roomName in ROOMS:
        room = hm.Room(env, roomName)
        room.add_device(dm.SmartLight(env, f'{roomName}_light'))
        home.add_room(room)

    home.rooms['livingroom'].add_device(dm.SmartTV(env, 'livingroom_tv'))

def day_divider(env: Environment) -> Generator[simpy.Event, None, None]:
    '''Prints the current day every 24 hours'''
    while True:
        timeslot = env.timeslot
        print(f'DAY {timeslot.Day}.{timeslot.Month}.{timeslot.Year} ----------------------------------------------')
        yield env.timeout(60*24)


if __name__ == '__main__':
    for i in range(SIM_ITERATIONS):
        print(f'##################### ITERATION {i+1} ####################')
        
        # Environment
        env = Environment(SIM_START)
        setup_home(env.home)

        # Day dividing prints
        env.process(day_divider(env))
        
        # Inhabitants
        stateLoggers: List[StateLogger] = []
        for i in range(1, NUM_OF_INHABITANTS + 1):
            inhabitant = StochasticInhabitant(env, str(i))
            inhabitant.location = env.home.rooms['bedroom'] # Start in the bedroom
            env.process(inhabitant.behaviour())
            
            # State logger
            logFilePath = f'./logs/inhabitant_{str(i)}-{SIM_ITERATIONS}iters.csv'
            stateLoggers.append(StateLogger(env, LOG_TIME_INTERVAL, logFilePath, inhabitant))
            env.eventHandler.subscribe('light_turned_on', stateLoggers[-1].deviceTurnedOnHandler)
            env.eventHandler.subscribe('light_turned_off', stateLoggers[-1].deviceTurnedOffHandler)
            env.process(stateLoggers[-1].logBehavior())
        
        env.run(SIM_END)

        print('Finish time: ' + str(env.timeslot))
        print()
        for stateLogger in stateLoggers:
            stateLogger.close()