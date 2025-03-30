import argparse
from typing import Generator, List

import simpy

from . import deviceModels as dm
from . import homeModel as hm
from .constants import ROOMS
from .environment import Environment, Time
from .inhabitants.deterministic import DeterministicInhabitant
from .inhabitants.stochastic import StochasticInhabitant
from .stateLogger import EventStateLogger, PeriodicStateLogger, StateLogger

SIM_START = Time(Minute=0, Hour=0, Day=1, Month=1, Year=1).to_minutes()
SIM_END = Time(Minute=0, Hour=0, Day=1, Month=2, Year=1).to_minutes()

LOG_TIME_INTERVAL = 5  # minutes # Log every LOG_TIME_INTERVAL minutes

def setup_home(env: Environment):
    for roomName in ROOMS:
        room = hm.Room(env, roomName)
        room.add_device(dm.SmartLight(env, f"{roomName}_light"))
        env.home.add_room(room)

    env.home.rooms["livingroom"].add_device(dm.SmartTV(env, "livingroom_tv"))


def day_divider(env: Environment) -> Generator[simpy.Event, None, None]:
    """Prints the current day every 24 hours"""
    while True:
        timeslot = env.timeslot
        print(f"DAY {timeslot.Day}.{timeslot.Month}.{timeslot.Year} ----------------------------------------------")
        yield env.timeout(60 * 24)


def main():
    # Command line arguments parsing
    parser = argparse.ArgumentParser(description="Run the Simulation without MAS.")
    parser.add_argument("-n", "--num_inhabitants", type=int, default=1, help="The number of inhabitants in the simulation")
    parser.add_argument("-d", "--deterministic", action="store_true", help="Use deterministic inhabitant")
    parser.add_argument("--no_weekend", action="store_true", help="Inhabitant uses only workday behavior (no weekend behavior)")
    args = parser.parse_args()
    num_inhabitants= args.num_inhabitants
    deterministic = args.deterministic
    no_weekend = args.no_weekend
    assert num_inhabitants > 0

    # Environment
    env = Environment(SIM_START)
    setup_home(env)

    # Day dividing prints
    env.process(day_divider(env))

    # Inhabitants
    stateLoggers: List[StateLogger] = []
    for i in range(1, num_inhabitants + 1):
        inhabitant: StochasticInhabitant | DeterministicInhabitant
        if deterministic:
             inhabitant = DeterministicInhabitant(env, str(i), no_weekend)
        else:
            inhabitant = StochasticInhabitant(env, str(i), no_weekend)
        inhabitant.location = env.home.rooms["bedroom"]  # Start in the bedroom
        env.process(inhabitant.behaviour())

        # Periodic State logger
        logFilePath = f"./logs/inhabitant_{str(i)}.csv"
        logger = PeriodicStateLogger(env, LOG_TIME_INTERVAL, logFilePath, inhabitant)
        env.eventHandler.subscribe("light_turned_on", logger.deviceTurnedOnHandler)
        env.eventHandler.subscribe("light_turned_off", logger.deviceTurnedOffHandler)
        env.process(logger.logBehavior())
        stateLoggers.append(logger)

        # Event State Logger
        logFilePath = f"./logs/inhabitant_{str(i)}-events.csv"
        logger = EventStateLogger(env, logFilePath, inhabitant)
        env.eventHandler.subscribe("light_turned_on", logger.deviceTurnedOnHandler)
        env.eventHandler.subscribe("light_turned_off", logger.deviceTurnedOffHandler)
        env.eventHandler.subscribe("inhabitant_changed_location", logger.inhabitantChangedLocation)
        stateLoggers.append(logger)

    env.run(SIM_END)

    print(f"Finish time: {str(env.timeslot)}")
    print()
    for stateLogger in stateLoggers:
            stateLogger.close()


if __name__ == "__main__":
    main()
