"""
Author: Martin Zelenák (xzelen27@stud.fit.vutbr.cz)
Description: This script runs the smart home simulation without the multi-agent system.
Date: 2025-05-14
"""


import yaml
import argparse
from typing import Generator, List, Any

import simpy

from Simulation import deviceModels as dm
from Simulation import homeModel as hm
from Simulation.constants import ROOMS
from Simulation.environment import Environment, Time
from Simulation.inhabitants.deterministic import DeterministicInhabitant
from Simulation.inhabitants.stochastic import StochasticInhabitant
from Simulation.stateLogger import EventStateLogger, PeriodicStateLogger, StateLogger

# Load config
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run simulation without MAS")
    parser.add_argument("--config", "-c", default="config.yaml", 
                      help="Path to the configuration file (default: config.yaml)")
    return parser.parse_args()


Config: Any = None
args = parse_arguments()
config_path = args.config

try:
    with open(config_path, "r") as file:
        Config = yaml.safe_load(file)
    if not Config:
        print(f"Failed to load config file: {config_path} (empty file)")
        exit(1)
except FileNotFoundError:
    print(f"Config file not found: {config_path}")
    exit(1)
except Exception as e:
    print(f"Error loading config file: {config_path} - {str(e)}")
    exit(1)

LOG_TIME_INTERVAL = Config["Simulation"]["log_interval"]  # minutes # Log every LOG_TIME_INTERVAL minutes
NUM_OF_STOCHASTIC_INHABITANTS = Config["Simulation"]["stochastic_inhabitants"]
NUM_OF_DETERMINISTIC_INHABITANTS = Config["Simulation"]["deterministic_inhabitants"]
NO_WEEKEND = Config["Simulation"].get("no_weekend", False)

start_dict = Config["Simulation"]["start"]
end_dict = Config["Simulation"]["end"]
SIM_START = Time(Minute=start_dict["minute"], 
                 Hour=start_dict["hour"], 
                 Day=start_dict["day"], 
                 Month=start_dict["month"], 
                 Year=start_dict["year"]).to_minutes()
SIM_END = Time(Minute=end_dict["minute"], 
               Hour=end_dict["hour"], 
               Day=end_dict["day"], 
               Month=end_dict["month"], 
               Year=end_dict["year"]).to_minutes()

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
    # Environment
    env = Environment(SIM_START)
    setup_home(env)

    # Day dividing prints
    env.process(day_divider(env))

    stateLoggers: List[StateLogger] = []
    logFolder = Config["Simulation"]["inhabitants_logs"]["folder"]
    
    # Deterministic inhabitants
    for i in range(1, NUM_OF_DETERMINISTIC_INHABITANTS + 1):
        inhabitant = DeterministicInhabitant(env, str(i), NO_WEEKEND)
        inhabitant.location = env.home.rooms["bedroom"]  # Start in the bedroom
        env.process(inhabitant.behavior())

        # Periodic State logger
        logFilePath = f"{logFolder}/inhabitant_{str(i)}-{LOG_TIME_INTERVAL}min.csv"
        logger = PeriodicStateLogger(env, LOG_TIME_INTERVAL, logFilePath, inhabitant)
        env.eventHandler.subscribe("light_turned_on", logger.deviceTurnedOnHandler)
        env.eventHandler.subscribe("light_turned_off", logger.deviceTurnedOffHandler)
        env.process(logger.logBehavior())
        stateLoggers.append(logger)

        # Event State Logger
        logFilePath = f"{logFolder}/inhabitant_{str(i)}-events.csv"
        logger = EventStateLogger(env, logFilePath, inhabitant)
        env.eventHandler.subscribe("light_turned_on", logger.deviceTurnedOnHandler)
        env.eventHandler.subscribe("light_turned_off", logger.deviceTurnedOffHandler)
        env.eventHandler.subscribe("inhabitant_changed_location", logger.inhabitantChangedLocation)
        stateLoggers.append(logger)
    
    # Stochastic inhabitants
    for i in range(NUM_OF_DETERMINISTIC_INHABITANTS + 1, NUM_OF_DETERMINISTIC_INHABITANTS + NUM_OF_STOCHASTIC_INHABITANTS + 1):
        inhabitant = StochasticInhabitant(env, str(i), NO_WEEKEND)
        inhabitant.location = env.home.rooms["bedroom"]  # Start in the bedroom
        env.process(inhabitant.behavior())

        # Periodic State logger
        logFilePath = f"{logFolder}/inhabitant_{str(i)}-{LOG_TIME_INTERVAL}min.csv"
        logger = PeriodicStateLogger(env, LOG_TIME_INTERVAL, logFilePath, inhabitant)
        env.eventHandler.subscribe("light_turned_on", logger.deviceTurnedOnHandler)
        env.eventHandler.subscribe("light_turned_off", logger.deviceTurnedOffHandler)
        env.process(logger.logBehavior())
        stateLoggers.append(logger)

        # Event State Logger
        logFilePath = f"{logFolder}/inhabitant_{str(i)}-events.csv"
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
