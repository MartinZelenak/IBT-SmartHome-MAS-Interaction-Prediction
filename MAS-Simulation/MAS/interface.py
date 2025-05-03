import logging
import threading
import time
from multiprocessing import Process
from typing import Callable, Dict, Optional, Tuple

import xmpp
from xmpp import JID

from .config import *
from .data import *
from .messages.interfaceMessages import *
from .system import system_start

logger = logging.getLogger('MAS.interface')

class Interface():
    '''The system interface class.
    This class is used to communicate with the multi-agent system.
    It connects to the XMPP server and sends and receives messages to and from the system agents.
    It creates a new thread to process incoming messages.
    It starts and stops the system as a subprocess. '''
    def __init__(self, interface_jid: JID, interface_password: str, system_jid: JID, system_password: str, use_SSL: bool = False):
        '''Initializes the system interface.
        Connects to the XMPP server and logs in with the given JID and password.
        
        Args:
            interface_jid: The JID of the interface client.
            interface_password: The password of the interface client.
            system_jid: The JID of the system main agent.
            system_password: The password of the system main agent.'''
        if not isinstance(interface_jid, JID):
            raise ValueError('Invalid interface JID value')
        if not isinstance(interface_password, str):
            raise ValueError('Invalid interface password value')
        if not isinstance(system_jid, JID):
            raise ValueError('Invalid system JID value')
        if not isinstance(system_password, str):
            raise ValueError('Invalid system password value')
        
        self.jid: str = str(interface_jid)
        self.password: str = interface_password
        self.system_jid: str = str(system_jid)
        self.system_password: str = system_password

        self.environment_state = State()
        self.system_process: Process | None = None
        self.main_agent_ready: bool = False
        self.user_agents: List[str] = []
        self.device_agents: List[str] = []
        self.pending_actions: Dict[str, Tuple[TimeSlot, int|float]] = {}

        # Connect to the XMPP server
        self._client: xmpp.Client | None = None
        self._use_SSL: bool = use_SSL
        self._process_incoming_messages: bool = False
        self._expected_replies: Dict[str, xmpp.Message | None] = {}
        self._connect()

    def _connect(self):
        jid = xmpp.JID(self.jid)
        self._client = xmpp.Client(server=jid.getDomain(), debug=None)
        if not self._client.connect():
            raise Exception('Could not connect to XMPP server')
        if not self._client.auth(jid.getNode(), self.password):
            raise Exception('Could not authenticate with XMPP server')
        logger.info('Interface connected to XMPP server')

        self._client.RegisterDisconnectHandler(lambda: print('Interface disconnected from XMPP server'))
        self._client.RegisterHandler('message', self._handle_message) # type: ignore
        
        self._client.sendInitPresence(requestRoster=0)

        # Messsage processing thread
        self._process_incoming_messages = True
        def process_incoming_messages_thread():
            while self._process_incoming_messages:
                self._client.Process(timeout=1) # type: ignore
        self._messsage_processing_thread = threading.Thread(target=process_incoming_messages_thread, name='MessageProcessingThread', daemon=True)
        self._messsage_processing_thread.start()

    def _handle_message(self, conn, msg: xmpp.Message):
        '''Handle incoming messages.'''
        logger.debug(f'Message from {msg.getFrom()}: {msg.getBody()}')
        if msg.getType() not in ['error', 'chat', None]:
            logger.info(f'Unknown message type: {msg.getType()}')
            return

        metadata = get_spade_metadata_from_xmpp_message(msg)

        # Check if the message is a reply to a sent message
        in_reply_to = metadata.get('in-reply-to') if metadata else None
        if in_reply_to:
            if in_reply_to in self._expected_replies:
                self._expected_replies[in_reply_to] = msg
                logger.debug(f'Expected reply message from {msg.getFrom()}: {msg.getBody()}')
                return
            else:
                logger.debug(f'Unexpected reply message from {msg.getFrom()}: {msg.getBody()}')

        # Match the message to a message type
        if msg.getType() == 'error':
            logger.warning(f'XMPP error message: {msg.getBody()}')
        elif ErrorMessage().match(msg):
            logger.error(f'System error message: {msg.getBody()}')
        elif AgentReadyMessage().match(msg):
            if msg.getBody() == self.system_jid:
                self.main_agent_ready = True
                logger.debug('Main agent ready message')
            else:
                logger.debug('Agent ready message')
        elif ActionMessage().match(msg):
            action_msg: ActionMessage = ActionMessage(msg.getBody(), msg.getFrom())
            action = action_msg.Action
            if action is not None:
                self.pending_actions[action[1]] = (action[0], action[2])
                logger.debug(f'Action message: {action}')
            else:
                logger.warning('Received an action message without action!')
        else:
            logger.debug(f'Unmatched message: {msg.getBody()}')

    def _send_message(self, message: Message):
        if not self.system_process or not self.system_process.is_alive():
            raise Exception('System process is not running')
        
        if self._client is None:
            raise Exception('Interface not connected!')

        if message.get_spade_metadata('reply-with') is not None:
            raise ValueError('Message cannot have a "reply-with" metadata')

        self._client.send(message) # type: ignore

    def _send_message_and_wait_for_reply(self, message: Message, timeout: int = 25) -> xmpp.Message|None:
        if not self.system_process or not self.system_process.is_alive():
            raise Exception('System process is not running')

        expected_response_id: str = message.get_spade_metadata('reply-with') # type: ignore
        if expected_response_id is None:
            raise ValueError('Message does not have a "reply-with" metadata')

        self._expected_replies[expected_response_id] = None
        self._client.send(message) # type: ignore

        abort_time = time.time() + timeout
        while time.time() < abort_time:
            if self._expected_replies[expected_response_id] is not None:
                return self._expected_replies.pop(expected_response_id)
            time.sleep(0.5)
        return None

    def start(self, 
              *, 
              timeout: int = 25, 
              log_conf: Optional[dict|int] = None, 
              prediction_conf: Optional[PredictionConfig] = None, 
              get_time_func: Optional[Callable[[], TimeSlot]] = None, 
              **kwargs):
        '''Start the system process.
        
        Args:
            timeout: The time to wait for the main agent to be ready.
            log_conf: The logging level or logging configuration.
            prediction_conf: The prediction configuration.
            get_time_func: The function to get the current time. Default is get_time().
            kwargs: Additional arguments to pass to the system.
                - get_time_func_params: Iterable with arguments to pass to the get_time_func function.'''
        if self.system_process and self.system_process.is_alive():
            raise Exception('System process is already running')
        
        if prediction_conf is not None and not isinstance(prediction_conf, PredictionConfig):
            raise ValueError('Invalid prediction configuration value')
        
        system_kwargs = {
            'interface_jid': self.jid, 
            'main_agent_jid': self.system_jid, 
            'main_agent_password': self.system_password, 
            'log_conf': log_conf, 
            'prediction_conf': prediction_conf, 
            'get_time_func': get_time_func
        }
        if 'get_time_func_params' in kwargs:
            system_kwargs['get_time_func_params']=kwargs['get_time_func_params']

        self.system_process = Process(target=system_start, 
                                      kwargs=system_kwargs, 
                                      name='Multi-Agent System', daemon=True)
        self.system_process.start()
        if not self.system_process.is_alive():
            raise Exception('System process did not start')
        
        # Wait for the main agent to be ready
        abort_time = time.time() + timeout
        while time.time() < abort_time:
            if self.main_agent_ready:
                return
            time.sleep(1)
        raise Exception('No response from main agent')

    def add_user(self, jid: JID, password: str, initial_location: Optional[int] = None, timeout: int = 25) -> bool:
        '''Add a new user agent to the system.

        Args:
            jid: The JID of the new user agent.
            password: The password of the new user agent.
            initial_location: The initial location of the new user agent. If None, user's location won't be a part of the environment state.'''
        if not isinstance(jid, JID):
            raise ValueError('Invalid JID value')
        if not isinstance(password, str):
            raise ValueError('Invalid password value')
        if initial_location is not None and len(self.environment_state.DeviceStates) != 0:
                raise Exception('Cannot add new user with location after device agents have been added. This would change the size of the state vector for already learning device agents.')

        msg = AddNewUserAgentMessage(to=self.system_jid, jid=str(jid), password=password)
        response = self._send_message_and_wait_for_reply(msg, timeout)

        if response is None:
            logger.error('No response from system')
            return False
        if ErrorMessage().match(response):
            logger.error(f'Error adding user: {response.getBody()}')
            return False
        
        self.user_agents.append(str(jid))
        if initial_location is not None:
            self.environment_state.UserLocations[str(jid)] = initial_location
        return True

    def user_set_device_filter(self, 
                               user_jid: JID, 
                               device_jid: JID, 
                               enabled: Optional[bool], 
                               threshold_off: Optional[float] = None, 
                               threshold_on: Optional[float] = None):
        '''Set a device filter for taking actions based on prediction for a given user.
        
        Args:
            user_jid: The JID of the user agent.
            device_jid: The JID of the device agent.
            enabled: Enable or disable the filter. Disabled => no actions for this device. Keep current if None (Default: Disabled)
            treshold_off: The treshold value for the filter to turn off the device. Prediction < treshold_off => turn off. Keep current if None (Default: 0.5)
            threshold_on: The treshold value for the filter to turn on the device. Prediction > threshold_on => turn on. Keep current if None (Default: 0.5)'''
        if not isinstance(user_jid, JID):
            raise ValueError('Invalid user JID value')
        if not isinstance(device_jid, JID):
            raise ValueError('Invalid device JID value')
        if enabled is not None and not isinstance(enabled, bool):
            raise ValueError('Invalid enabled value')
        if threshold_off is not None and not isinstance(threshold_off, (float, int)):
            raise ValueError('Invalid treshold_off value')
        if threshold_on is not None and not isinstance(threshold_on, (float, int)):
            raise ValueError('Invalid threshold_on value')
        if enabled is None and threshold_off is None and threshold_on is None:
            raise ValueError('No filter settings provided')
        if str(user_jid) not in self.user_agents:
            raise ValueError('Provided user JID does not exist in the system')
        if str(device_jid) not in self.device_agents:
            raise ValueError('Provided device JID does not exist in the system')

        settings = DeviceFilter(Device_JID=str(device_jid), Enabled=enabled, Treshold_Off=threshold_off, Treshold_On=threshold_on)
        msg = SetDeviceFilterMessage(to=str(user_jid), filter_settings=settings)
        
        response = self._send_message_and_wait_for_reply(msg)

        if response is None:
            logger.error('No response from system')
            return False
        if ErrorMessage().match(response):
            logger.error(f'Error setting device filter: {response.getBody()}')
            return False
        return True

    def add_device(self, jid: JID, password: str, initial_state: float|int, timeout: int = 25) -> bool:
        '''Add a new device agent to the system.'''
        if not isinstance(jid, JID):
            raise ValueError('Invalid JID value')
        if not isinstance(password, str):
            raise ValueError('Invalid password value')
        if not isinstance(initial_state, (float, int)):
            raise ValueError('Invalid initial state value')

        msg = AddNewDeviceAgentMessage(to=self.system_jid, jid=str(jid), password=password)
        response = self._send_message_and_wait_for_reply(msg, timeout)

        if response is None:
            logger.error('No response from system')
            return False
        if ErrorMessage().match(response):
            logger.error(f'Error adding device: {response.getBody()}')
            return False
        
        self.device_agents.append(str(jid))
        self.environment_state.DeviceStates[str(jid)] = initial_state
        return True

    def update_state(self, 
                     user_locations: Optional[Dict[str, int]] = None, 
                     device_states: Optional[Dict[str, float|int]] = None):
        '''Send a new state to the system.'''
        if user_locations is None and device_states is None:
            return

        if user_locations:
            for jid, location in user_locations.items():
                if jid not in self.environment_state.UserLocations:
                    raise ValueError(f'User {jid} not found in the environment state')
                self.environment_state.UserLocations[jid] = location

        if device_states:
            for jid, state in device_states.items():
                if jid not in self.environment_state.DeviceStates:
                    raise ValueError(f'Device {jid} not found in the environment state')
                self.environment_state.DeviceStates[jid] = state

        msg = NewStateMessage(to=self.system_jid, state=self.environment_state)
        self._send_message(msg)

    def update_state_with_response(self, 
                                   user_locations: Optional[Dict[str, int]] = None, 
                                   device_states: Optional[Dict[str, float|int]] = None, 
                                   timeout: int = 25) -> bool:
        '''Send a new state to the system and wait for a response.'''
        if user_locations is None and device_states is None:
            return True

        if user_locations:
            for jid, location in user_locations.items():
                if jid not in self.environment_state.UserLocations:
                    raise ValueError(f'User {jid} not found in the environment state')
                self.environment_state.UserLocations[jid] = location

        if device_states:
            for jid, state in device_states.items():
                if jid not in self.environment_state.DeviceStates:
                    raise ValueError(f'Device {jid} not found in the environment state')
                self.environment_state.DeviceStates[jid] = state

        msg = NewStateMessage(to=self.system_jid, state=self.environment_state)
        msg.expect_reply(f'new-state-{msg.getID()}')
        response =  self._send_message_and_wait_for_reply(msg, timeout)

        if response is None:
            logger.error('No response from system')
            return False
        if ErrorMessage().match(response):
            logger.error(f'Error updating state: {response.getBody()}')
            return False
        return True

    def trigger_predictions(self):
        '''Trigger the prediction process for all devices in the system.'''
        msg = TriggerPredictionMessage(to=self.system_jid)
        self._send_message(msg)

    def trigger_predictions_with_response(self, timeout: int = 25) -> bool:
        '''Trigger the prediction process for all devices in the system and wait for a response.'''
        msg = TriggerPredictionMessage(to=self.system_jid)
        msg.expect_reply(f'trigger-prediction-{msg.getID()}')
        response =  self._send_message_and_wait_for_reply(msg, timeout)

        if response is None:
            logger.error('No response from system')
            return False
        if ErrorMessage().match(response):
            logger.error(f'Error triggering predictions: {response.getBody()}')
            return False
        return True
    
    def send_stop_message(self, to: str):
        '''Send a stop message to the given agent.'''
        msg = StopMessage(to=to)
        self._send_message(msg)
    
    def pending_actions_pop(self) -> Tuple[TimeSlot, str, int|float]|None:
        '''Get an action to be taken.
        Returns a tuple of TimeSlot, device JID and desired state.'''
        if len(self.pending_actions) == 0:
            return None
        device_jid, action = self.pending_actions.popitem()
        return (action[0], device_jid, action[1])

    def stop(self):
        '''Sends stop message and terminates the system process.'''
        if self.system_process is None:
            logger.warning('System process terminated before stop() was called')
            return
        if self.system_process.is_alive():
            self.send_stop_message(self.system_jid)
            self.system_process.join(timeout=25)
            if self.system_process.is_alive():
                self.system_process.terminate()
                self.system_process.join(timeout=25)
                logger.info('System process terminated')
            if self.system_process.is_alive():
                self.system_process.kill()
                self.system_process.join()
                logger.warning('System process forcefuly killed')
        self.system_process.close()
        self.system_process = None

    def __del__(self):
        '''Terminates the system process if it is still running.'''
        if hasattr(self, 'system_process') and self.system_process and self.system_process.is_alive():
            self.stop()
        if hasattr(self, '_messsage_processing_thread') and self._messsage_processing_thread.is_alive():
            self._process_incoming_messages = False
            self._messsage_processing_thread.join()