import logging

import MAS


def get_time() -> MAS.TimeSlot:
    import datetime
    now = datetime.datetime.now()
    return MAS.TimeSlot(now.minute, now.hour, now.weekday())

def main():
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s: %(message)s')

    system_interface: MAS.Interface= MAS.Interface(interface_jid=MAS.JID('interface@localhost'), 
                                                        interface_password='password',
                                                        system_jid=MAS.JID('mainagent@localhost'),
                                                        system_password='password')
    system_interface.start(timeout=10, log_conf=logging.INFO, get_time_func=get_time)        

    # Initial agents
    added_initial_agents = False
    if input('# Add initial agents? (y/n) \n') == 'y':
        system_interface.add_user(MAS.JID('useragent@localhost'), 
                                  'password', 
                                  initial_location=1)
        system_interface.add_device(MAS.JID('deviceagent@localhost'), 
                                    'password', 
                                    initial_state=0)
        added_initial_agents = True

    if input('# Set device filter? (y/n) \n') == 'y':
        enabled = input('## Enable filter? (y/n) \n') == 'y'
        treshold_off = float(input('## Enter treshold off: '))
        treshold_on = float(input('## Enter treshold on: '))
        system_interface.user_set_device_filter(MAS.JID('useragent@localhost'), 
                                                MAS.JID('deviceagent@localhost'), 
                                                enabled, 
                                                treshold_off, 
                                                treshold_on)

    # Add a new user
    while input('# Add new user? (y/n) \n') == 'y':
        jid = input('## Enter the user MAS.JID: ')
        password = input('## Enter the user password: ')
        system_interface.add_user(MAS.JID(jid), password, None if added_initial_agents else 1)

    # Add new device agent
    while input('# Add new device agent? (y/n) \n') == 'y':
        jid = input('## Enter the device agent MAS.JID: ')
        password = input('## Enter the device agent password: ')
        system_interface.add_device(MAS.JID(jid), password, 0)

    print(f'# State: {system_interface.environment_state}')

    # Send state update
    while input('# Send state update? (y/n) \n') == 'y':
        system_interface.update_state({'useragent@localhost': 2}, {'deviceagent@localhost': 1})
        system_interface.trigger_predictions()

        # Actions
        # action = system_interface.pending_actions_pop()
        # while action is not None:
        #     print(f'Action: {action}')
        #     action = system_interface.pending_actions_pop()

    # Stop the system
    system_interface.stop()

    print('MAS.Interface finished')

if __name__ == '__main__':
    main()