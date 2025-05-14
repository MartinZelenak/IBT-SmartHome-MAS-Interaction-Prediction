"""
Author: Martin Zelenák (xzelen27@stud.fit.vutbr.cz)
Description: This script runs the multi-agent system and provides a CLI hands on demonstration
            of using the multi-agent system throught the system interface.
Date: 2025-05-14
"""


import logging
import logging.config
import yaml

import MAS


def get_time() -> MAS.TimeSlot:
    import datetime
    now = datetime.datetime.now()
    return MAS.TimeSlot(now.minute, now.hour, now.weekday())

def main():
    # Load config
    config = None
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    if not config:
        print("Failed to load config file")
        exit(1)
    
    # Log on debug level
    config["MAS"]["logging"]["loggers"]["MAS"]["level"] = "DEBUG"
    logging.config.dictConfig(config["MAS"]["logging"])
    
    # Start the system
    system_interface: MAS.Interface= MAS.Interface(interface_jid=MAS.JID("interface@localhost"), 
                                                   interface_password="password",
                                                   system_jid=MAS.JID("mainagent@localhost"),
                                                   system_password="password")
    system_interface.start(timeout=10,
                           log_conf=config["MAS"]["logging"],
                           prediction_conf=MAS.config.PredictionConfig.from_dict(config["MAS"]["prediction"]),
                           get_time_func=get_time)

    # Initial agents
    added_initial_agents = False
    if input("# Add initial agents? (y/n) \n") == "y":
        system_interface.add_user(MAS.JID("useragent@localhost"), 
                                  "password", 
                                  initial_location=1)
        system_interface.add_device(MAS.JID("deviceagent@localhost"), 
                                    "password", 
                                    initial_state=0)
        added_initial_agents = True

    if added_initial_agents and input("# Set initial device filter? (y/n) \n") == "y":
        enabled = input("## Enable filter? (y/n) \n") == "y"
        treshold_off = float(input("## Enter treshold off: "))
        treshold_on = float(input("## Enter treshold on: "))
        system_interface.user_set_device_filter(MAS.JID("useragent@localhost"), 
                                                MAS.JID("deviceagent@localhost"), 
                                                enabled, 
                                                treshold_off, 
                                                treshold_on)

    # Add a new user
    while input("# Add new user? (y/n) \n") == "y":
        jid = input("## Enter the user JID: ")
        password = input("## Enter the user password: ")
        system_interface.add_user(MAS.JID(jid), password, None if added_initial_agents else 1)

    # Add new device agent
    while input("# Add new device agent? (y/n) \n") == "y":
        jid = input("## Enter the device agent JID: ")
        password = input("## Enter the device agent password: ")
        system_interface.add_device(MAS.JID(jid), password, 0)

    while input("# Set new or update a user agent device filter? (y/n) \n") == "y":
        user_agent_jid = input("## Enter the user agent JID: ")
        device_agent_jid = input("## Enter the device agent JID: ")
        enabled = input("## Enable filter? (y/n) \n") == "y"
        treshold_off = float(input("## Enter treshold off: "))
        treshold_on = float(input("## Enter treshold on: "))
        system_interface.user_set_device_filter(MAS.JID(user_agent_jid), 
                                                MAS.JID(device_agent_jid), 
                                                enabled, 
                                                treshold_off, 
                                                treshold_on)

    print(f"# State: {system_interface.environment_state}")

    # Send state update
    while input("# Send state update? (y/n) \n") == "y":
        # Change state
        if added_initial_agents:
            user_location = system_interface.environment_state.UserLocations["useragent@localhost"]
            device_state = system_interface.environment_state.DeviceStates["deviceagent@localhost"]
            system_interface.update_state(
                {"useragent@localhost": 2 if user_location == 1 else 1},
                {"deviceagent@localhost": 1 if device_state == 0 else 0},
            )
        else:
            print("# Missing the intial agents. Not updating state and only triggering prediction.")
        
        #  Trigger predictions
        system_interface.trigger_predictions()

        # Actions
        action = system_interface.pending_actions_pop()
        while action is not None:
            print(f"Action: {action}")
            action = system_interface.pending_actions_pop()

    # Stop the system
    system_interface.stop()

    print("MAS.Interface finished")

if __name__ == "__main__":
    main()
