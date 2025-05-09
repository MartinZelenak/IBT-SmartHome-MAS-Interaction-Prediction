import logging
import logging.config
import multiprocessing
import multiprocessing.sharedctypes
import os
import pickle
import socket
import time
import argparse
from typing import Dict, Generator, List, Any

import simpy
import yaml

import MAS.messages.spadeMessages as spadeMessages
import MAS.system
import Simulation.deviceModels as dm
import Simulation.homeModel as hm
from Simulation.constants import ROOMS
from Simulation.environment import Environment, Time
from Simulation.inhabitants.stochastic import StochasticInhabitant
from Simulation.stateLogger import PeriodicStateLogger

# Load config
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run simulation with MAS")
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

LOG_TIME_INTERVAL = Config["Simulation"]["log_interval"] # minutes # Log every LOG_TIME_INTERVAL minutes
NUM_OF_STOCHASTIC_INHABITANTS = Config["Simulation"]["stochastic_inhabitants"]
NUM_OF_DETERMINISTIC_INHABITANTS = Config["Simulation"]["deterministic_inhabitants"]
MAS_JABBER_HOST = Config["Simulation"]["jabber_host"]

start_dict = Config["Simulation"]["start"]
end_dict = Config["Simulation"]["end"]
SIM_START = Time(Minute=start_dict["minute"], 
                     Hour=start_dict["hour"], 
                     Day=start_dict["day"], 
                     Month=start_dict["month"], 
                     Year=start_dict["year"])
SIM_END   = Time(Minute=end_dict["minute"], 
                     Hour=end_dict["hour"], 
                     Day=end_dict["day"], 
                     Month=end_dict["month"], 
                     Year=end_dict["year"])

# Logging
logging.config.dictConfig(Config["MAS"]["logging"])
PredictionConfig = MAS.PredictionConfig.from_dict(Config["MAS"]["prediction"])

JID_device_dict: Dict[str, dm.SmartDevice] = {}

def setup_home(env: Environment):
    for roomName in ROOMS:
        room = hm.Room(env, roomName)
        device = dm.SmartLight(env, f"{roomName}_light")
        JID_device_dict[f"{device.name}@{MAS_JABBER_HOST}"] = device
        room.add_device(device)
        env.home.add_room(room)

    device = dm.SmartTV(env, "livingroom_tv")
    JID_device_dict[f"{device.name}@{MAS_JABBER_HOST}"] = device
    env.home.rooms["livingroom"].add_device(device)

# MAS - Subprocess and communication socket
MAS_subprocess = None
MAS_socket = None
def MAS_start():
    global MAS_subprocess
    global MAS_socket
    tcp_config = Config["Simulation"]["TCP_relay_agent"]

    system_kwargs = {
        "tcp_interface_jid": f"interface@{MAS_JABBER_HOST}", 
        "tcp_interface_password": "password", 
        "tcp_interface_host_ip": tcp_config["host_ip"], 
        "tcp_interface_port": tcp_config["host_port"], 
        "main_agent_jid": f"mainagent@{MAS_JABBER_HOST}", 
        "main_agent_password": "password", 
        "log_conf": Config["MAS"]["logging"], 
        "prediction_conf": PredictionConfig, 
        "get_time_func": MAS_get_time, 
        "get_time_func_params": [current_env_time]
    }

    MAS_subprocess = multiprocessing.Process(target=MAS.system.system_start_tcp, 
                                             kwargs=system_kwargs, 
                                             name="Multi-Agent System", daemon=False)
    MAS_subprocess.start()

    # Connect to the TCP relay agent
    print("Connecting to TCP relay agent...")
    MAS_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            MAS_socket.connect((tcp_config["host_ip"], tcp_config["host_port"]))
            break
        except Exception as e:
            print(f"Connection failed: {e}. \nRetrying...")
            time.sleep(1)
    print("Connected to TCP relay agent")

def MAS_send_message(message: spadeMessages.Message) -> spadeMessages.Message|None:
    """Send a message to the MAS and return the response if any.
    Waits for response if the message has a "reply-with" metadata.

    Args:
        message (spadeMessages.Message): The message to send

    Returns:
        spadeMessages.Message|None: The response message if any
    """
    global MAS_socket
    if not MAS_socket or MAS_socket.fileno() == -1:
        print("Failed to send message: MAS socket is not connected")
        exit(1)

    try:
        serialized_msg = pickle.dumps(message)
        msg_length = len(serialized_msg)
        MAS_socket.sendall(msg_length.to_bytes(4, byteorder="big"))
        MAS_socket.sendall(serialized_msg)
    except Exception as e:
        print(f"Failed to send message: {e}")
        exit(1)

    if message.get_metadata("reply-with"):
        try:
            response_len = int.from_bytes(MAS_socket.recv(4), byteorder="big")
            response_bytes = MAS_socket.recv(response_len)
            return pickle.loads(response_bytes)
        except Exception as e:
            print(f"Failed to receive response: {e}")
            return None

    return None

def MAS_receive_message() -> spadeMessages.Message:
    """Wait for a message from the MAS

    Returns:
        spadeMessages.Message: The received message
    """
    global MAS_socket
    if not MAS_socket or MAS_socket.fileno() == -1:
        print("Failed to receive message: MAS socket is not connected")
        exit(1)

    try:
        msg_length = int.from_bytes(MAS_socket.recv(4), byteorder="big")
        msg_bytes = MAS_socket.recv(msg_length)
        return pickle.loads(msg_bytes)
    except Exception as e:
        print(f"Failed to receive message: {e}")
        exit(1)

def MAS_stop():
    global MAS_subprocess
    global MAS_socket
    if MAS_subprocess:
        print("Stopping MAS...")
        stop_msg = spadeMessages.StopMessage()
        stop_msg.to = f"mainagent@{MAS_JABBER_HOST}"
        MAS_send_message(stop_msg)

        stop_msg = spadeMessages.StopMessage()
        stop_msg.to = f"interface@{MAS_JABBER_HOST}"
        MAS_send_message(stop_msg)

        MAS_subprocess.join()
        if MAS_socket is not None:
            MAS_socket.close()

# MAS - Sharing simulation time with MAS subprocess
current_env_time = multiprocessing.Array("i", [int(SIM_START.Minute), SIM_START.Hour, SIM_START.day_of_week()], lock=True)
def MAS_get_time(current_env_time: multiprocessing.sharedctypes.SynchronizedArray) -> MAS.TimeSlot:
    """This method is used by the MAS to get the current simulation time."""
    # NOTE: current_env_time is a shared memory array 
    # and must be passed in as an argument
    # (cannot be accessed from the global scope)
    with current_env_time.get_lock():
        current_time = MAS.TimeSlot(current_env_time[0], current_env_time[1], current_env_time[2])
    return current_time
def MAS_update_time(env: Environment):
    """This method is used by the simulation to update the current time for the MAS"""
    global current_env_time
    with current_env_time.get_lock():
        timeslot = env.timeslot
        current_env_time[0] = int(timeslot.Minute)
        current_env_time[1] = timeslot.Hour
        current_env_time[2] = timeslot.day_of_week() - 1 # 0-indexed

def MAS_add_users(env: Environment) -> int:
    n_users = 0
    for inhabitant in env.inhabitans:
        print(f"Adding {inhabitant.name} user agent to MAS...")
        add_user_msg = spadeMessages.AddNewUserAgentMessage()
        add_user_msg.to = f"mainagent@{MAS_JABBER_HOST}"
        add_user_msg.JID = f"{inhabitant.name}@{MAS_JABBER_HOST}"
        add_user_msg.Password = "password"
        add_user_msg.set_metadata("reply-with", f"add-user-{inhabitant.name}")
        response = MAS_send_message(add_user_msg)
        if not isinstance(response, spadeMessages.SuccessMessage):
            print(f"Failed to add {inhabitant.name} user agent to MAS")
            print(f"Got response: {response}")
            exit(1)

        n_users += 1
    return n_users

def MAS_add_devices(env: Environment) -> int:
    n_devices = 0
    for room in env.home.rooms.values():
        for device in room.devices.values():
            if not isinstance(device, dm.SmartLight):
                continue
            # Add device agent to MAS
            print(f"Adding {device.name} device agent to MAS...")
            add_device_msg = spadeMessages.AddNewDeviceAgentMessage()
            add_device_msg.to = f"mainagent@{MAS_JABBER_HOST}"
            add_device_msg.JID = f"{device.name}@{MAS_JABBER_HOST}"
            add_device_msg.Password = "password"
            add_device_msg.set_metadata("reply-with", f"add-device-{device.name}")
            response = MAS_send_message(add_device_msg)
            if not isinstance(response, spadeMessages.SuccessMessage):
                print(f"Failed to add {device.name} device agent to MAS")
                print(f"Got response: {response}")
                exit(1)
            # Set filter for one user
            # NOTE: Device filter is set only for one user to reduce the number of messages
            # and to simplify the simulation
            inhabitant = env.inhabitans[0]
            prediction_filter = Config["Simulation"]["prediction_filter"]
            device_filter = spadeMessages.DeviceFilter(f"{device.name}@{MAS_JABBER_HOST}",
                                                        prediction_filter["enabled"],
                                                        prediction_filter["threshold_off"],
                                                        prediction_filter["threshold_on"])
            set_device_filter_msg = spadeMessages.SetDeviceFilterMessage()
            set_device_filter_msg.to = f"{inhabitant.name}@{MAS_JABBER_HOST}"
            set_device_filter_msg.FilterSettings = device_filter
            set_device_filter_msg.set_metadata("reply-with", f"set-filter-{device.name}-{inhabitant.name}")
            response = MAS_send_message(set_device_filter_msg)
            if not isinstance(response, spadeMessages.SuccessMessage):
                print(f"Failed to set filter for {device.name} device agent for {inhabitant.name} user agent")
                print(f"Got response: {response}")
                exit(1)
            n_devices += 1

    return n_devices

def MAS_handling(env: Environment, n_devices: int) -> Generator[simpy.Event, None, None]:
    # NOTE: To reduce the number of messages, the simulation will rely on the predict_on_new_state being True
    assert PredictionConfig.predict_on_new_state == True
    while True:
        # Update time
        MAS_update_time(env)
        
        # Update state
        user_locations = {}
        for inhabitant in env.inhabitans:
            if inhabitant.location:
                user_locations[f"{inhabitant.name}@{MAS_JABBER_HOST}"] = ROOMS.index(inhabitant.location.name)
        
        device_states = {}
        for room in env.home.rooms.values():
            for device in room.devices.values():
                if isinstance(device, dm.SmartLight):
                    device_states[f"{device.name}@{MAS_JABBER_HOST}"] = int(device.on)

        # Update state
        new_state_msg = spadeMessages.NewStateMessage()
        new_state_msg.to = f"mainagent@{MAS_JABBER_HOST}"
        new_state_msg.State = spadeMessages.State(user_locations, device_states)
        MAS_send_message(new_state_msg)

        # Wait for and take actions
        n_actions = 0
        while n_actions < n_devices:
            action_msg = MAS_receive_message()
            if isinstance(action_msg, spadeMessages.ActionMessage):
                n_actions += 1
                action = action_msg.Action
                if action is None:
                    continue
                # print(f"Action: {action}")
                device = JID_device_dict.get(action[1])
                if not isinstance(device, dm.SmartLight):
                    continue
                if action[2] == 0:
                    device.MAS_turn_off()
                elif action[2] == 1:
                    device.MAS_turn_on()

        yield env.timeout(Config["Simulation"]["MAS_update_interval"])

def day_divider(env: Environment) -> Generator[simpy.Event, None, None]:
    """Prints the current day every 24 hours"""
    while True:
        timeslot = env.timeslot
        print(f"DAY {timeslot.Day}.{timeslot.Month}.{timeslot.Year} ----------------------------------------------")
        yield env.timeout(60*24)

def MAS_print_device_stats():
    for device in JID_device_dict.values():
        if isinstance(device, dm.SmartLight):
            print(f"{device.name} - Correct: {device.MAS_correct_actions}, Incorrect: {device.MAS_incorrect_actions}")

def main():
    stateLoggers: List[PeriodicStateLogger] = []
    try:
        # Environment
        env = Environment(SIM_START.to_minutes())
        setup_home(env)

        # MAS start
        print("Starting MAS...")
        MAS_start()

        # Day dividing prints
        env.process(day_divider(env))

        # Inhabitants
        for i in range(NUM_OF_STOCHASTIC_INHABITANTS):
            inhabitant = StochasticInhabitant(env, f"inhabitant_{i}")
            inhabitant.location = env.home.rooms["bedroom"]  # Start in the bedroom
            env.process(inhabitant.behavior())

        for i in range(NUM_OF_DETERMINISTIC_INHABITANTS + len(env.inhabitans)):
            inhabitant = StochasticInhabitant(env, f"inhabitant_{i}")
            inhabitant.location = env.home.rooms["bedroom"]  # Start in the bedroom
            env.process(inhabitant.behavior())


        # State logger
        if Config["Simulation"]["inhabitants_logs"]["enabled"]:
            folder = Config["Simulation"]["inhabitants_logs"]["folder"]
            if os.path.exists(folder) == False:
                os.mkdir(folder)
            for inhabitant in env.inhabitans:
                logFilePath = os.path.join(folder, f"{inhabitant.name}.csv")
                stateLoggers.append(PeriodicStateLogger(env, LOG_TIME_INTERVAL, logFilePath, inhabitant))
                env.eventHandler.subscribe("light_turned_on", stateLoggers[-1].deviceTurnedOnHandler)
                env.eventHandler.subscribe("light_turned_off", stateLoggers[-1].deviceTurnedOffHandler)
                env.process(stateLoggers[-1].logBehavior())

        # MAS - Add users, devices and handling
        MAS_add_users(env)
        n_devices = MAS_add_devices(env)
        env.process(MAS_handling(env, n_devices))

        env.run(SIM_END.to_minutes())
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Stopping simulation...")
    except Exception as e:
        print(f"Error: {e}")

    print()
    print(f"Finish time: {str(env.timeslot)}")
    MAS_print_device_stats()
    print()
    MAS_stop()
    for stateLogger in stateLoggers:
        stateLogger.close()

if __name__ == "__main__":
    main()
