"""
Author: Martin Zelenák (xzelen27@stud.fit.vutbr.cz)
Description: The individual agents in the multi-agent system.
Date: 2025-05-14
"""


from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour, OneShotBehaviour
import asyncio
import logging
import os
from typing import List

from socket import socket, AF_INET, SOCK_STREAM
from select import select
import pickle

from .data import DeviceFilter

from .messages.spadeMessages import AddNewUserAgentMessage

from .config import PredictionConfig, default_prediction_config
from .data import *
from .messages.spadeMessages import *
from .predictionModel import PredictionModel
from .systemStats import SystemStats

logger = logging.getLogger('MAS.system')
prediction_config: PredictionConfig = default_prediction_config

def get_time() -> TimeSlot:
    '''Return the current time of the environment.
    This method should be set before running the system.'''
    raise NotImplementedError('The get time method is not set')

main_agent_jid: str | None = None
interface_jid: str | None = None

class BaseAgent(Agent):
    '''Base agent class with common behaviors and setup.'''
    class StopMessageBehavior(CyclicBehaviour):
        async def run(self):
            self.agent: BaseAgent
            msg = await self.receive(timeout=10)
            if msg:
                msg = StopMessage.from_message(msg)
                logger.debug(f'{self.agent.jid}: Stop message received with content: {msg.body}')

                if msg.get_metadata('reply-with'):
                    reply_msg = SuccessMessage(msg)
                    await self.send(reply_msg)
                await self.agent.stop()

    async def setup(self):
        logger.info(f'{self.jid}: Starting ...')

        stop_message_behavior = self.StopMessageBehavior()
        stop_message_template = StopMessage()
        self.add_behaviour(stop_message_behavior, stop_message_template)

    # NOTE: Stop is called twice in SPADE
    async def stop(self):
        if self.is_alive():
            logger.info(f'{self.jid}: Stopping ...')
        await super().stop()


class TCPRelayAgent(BaseAgent):
    def __init__(self, jid: str, password: str, host_ip: str, host_port: int, xmpp_port: int = 5222, verify_security: bool = False):
        super().__init__(jid, password, xmpp_port, verify_security)
        self.host_ip = host_ip
        self.host_port = host_port

    class WaitForConnectionBehavior(OneShotBehaviour):
        async def run(self):
            self.agent: TCPRelayAgent

            # Open socket for communication
            sockt = socket(AF_INET, SOCK_STREAM)
            sockt.bind((self.agent.host_ip, self.agent.host_port))
            sockt.listen(1)
            self.set('socket', sockt)

            # Wait for connection
            logger.info(f'{self.agent.jid} Listening on {self.agent.host_ip}:{self.agent.host_port}')
            poller_events, _, _ = select([sockt], [], [], 0)
            while not poller_events:
                await asyncio.sleep(1)
                poller_events, _, _ = select([sockt], [], [], 0)

            # Accept connection
            conn, addr = sockt.accept()
            logger.info(f'{self.agent.jid} Connected by {addr}')
            self.set('conn', conn)

            # Close welcome socket
            sockt.close()
            self.set('socket', None)

            # Start relaying messages
            self.agent.add_behaviour(self.agent.RelayIncommingMessagesBehavior(0.1))
            self.agent.add_behaviour(self.agent.SendOutgoingMessagesBehavior())

    class RelayIncommingMessagesBehavior(PeriodicBehaviour):
        async def run(self):
            self.agent: TCPRelayAgent

            # Get connection
            conn: socket = self.agent.get('conn')
            if not conn or conn.fileno() == -1:
                logger.warning(f'{self.agent.jid} TCP connection closed.')
                self.agent.add_behaviour(self.agent.WaitForConnectionBehavior())
                self.kill()
                return
            
            # Check for messages
            poller_events, _, _ = select([conn], [], [], 0)
            while not poller_events:
                await asyncio.sleep(0.1)
                poller_events, _, _ = select([conn], [], [], 0)

            # Receive message
            msg: Message | None = None
            try:
                msg_size = conn.recv(4)
                data = conn.recv(int.from_bytes(msg_size, byteorder='big'))

                msg = pickle.loads(data)
                logger.debug(f'{self.agent.jid} Message received from client: {msg}')
            except Exception as e:
                logger.error(f'{self.agent.jid} Error processing message: {e}')
            
            # Relay message
            if msg:
                msg.sender = str(self.agent.jid)
                await self.send(msg)

    class SendOutgoingMessagesBehavior(CyclicBehaviour):
        async def run(self):
            self.agent: TCPRelayAgent
            msg = await self.receive(timeout=10)
            if msg:
                logger.debug(f'{self.agent.jid} Sending message: {msg}')
                # Get connection
                conn: socket = self.agent.get('conn')
                # Send message
                if conn and conn.fileno() != -1:
                    try:
                        serialized_msg = pickle.dumps(msg)
                        msg_length = len(serialized_msg)
                        conn.sendall(msg_length.to_bytes(4, byteorder='big'))
                        conn.sendall(serialized_msg)
                    except Exception as e:
                        logger.error(f'{self.agent.jid} Error sending message: {e}')
                else:
                    logger.warning(f'{self.agent.jid} Connection closed. Cannot send message.')

    async def setup(self):
        await super().setup()

        self.add_behaviour(self.WaitForConnectionBehavior())

    async def stop(self):
        if sockt := self.get('socket'):
            sockt.close()
            self.set('socket', None)
        if conn := self.get('conn'):
            conn.close()
            self.set('conn', None)

        await super().stop()


class MainAgent(BaseAgent):
    '''Agent used as a system messenger, create new agents and stop created agents.
    When receiving a stop message, stop messages will be sent to all created agents and then this agent will stop'''
    def check_new_agent_jid(self, jid: str):
        if jid == str(self.jid):
            raise Exception('Cannot add a new agent with the same JID as the main agent')
        if jid in [str(agent.jid) for agent in self.get('device_agents')]:
            raise Exception('Device agent with the same JID already exists')
        if jid in [str(agent.jid) for agent in self.get('user_agents')]:
            raise Exception('User agent with the same JID already exists')

    class SendReadyMessageBehavior(OneShotBehaviour):
        async def run(self):
            self.agent: MainAgent
            msg = AgentReadyMessage()
            msg.to = str(interface_jid)
            msg.body = str(self.agent.jid)
            await self.send(msg)
            logger.debug(f'{self.agent.jid}: Ready message sent')

    class ReceiveAddNewUserAgentBehavior(CyclicBehaviour):
        '''Behavior to receive a message to add a new user. The message should contain the new agent's JID and password.'''
        async def run(self):
            self.agent: MainAgent
            msg = await self.receive(timeout=10)
            if msg:
                msg = AddNewUserAgentMessage.from_message(msg)
                logger.debug(f'{self.agent.jid}: Add new user received with content: {msg.body}')

                fail_text = ''
                try:
                    new_agent_jid, new_agent_password = msg.body.split(' ')
                    if not new_agent_jid or not new_agent_password:
                        raise Exception('Invalid JID or password')
                    
                    self.agent.check_new_agent_jid(new_agent_jid)

                    new_user_agent = UserAgent(new_agent_jid, new_agent_password)
                    await new_user_agent.start(auto_register=True)
                except Exception as e:
                    fail_text = str(e)
                    new_user_agent = None
                
                if not new_user_agent or not new_user_agent.is_alive():
                    logger.warning(f'{self.agent.jid}: Failed to add new user {new_agent_jid}. {fail_text}')
                    error_msg = ErrorMessage(msg)
                    error_msg.body = f'Failed to add new user. {fail_text}'
                    await self.send(error_msg)
                    return
                
                self.get('user_agents').append(new_user_agent)
                SystemStats().users_locations[new_agent_jid] = []
                
                reply_msg = SuccessMessage(msg)
                await self.send(reply_msg)
                logger.debug(f'{self.agent.jid}: New user agent {new_agent_jid} added')

    class ReceiveAddNewDeviceAgentBehavior(CyclicBehaviour):
        '''Behavior to receive a message to add a new device. The message should contain the new agent's JID and password.'''
        async def run(self):
            self.agent: MainAgent
            msg = await self.receive(timeout=10)
            if msg:
                msg = AddNewDeviceAgentMessage.from_message(msg)
                logger.debug(f'{self.agent.jid}: Add new device agent received with content: {msg.body}')

                new_device_agent = None
                fail_text = ''
                try:
                    new_agent_jid, new_agent_password = msg.body.split(' ')
                    if not new_agent_jid or not new_agent_password:
                        raise Exception('Invalid JID or password')
                    
                    self.agent.check_new_agent_jid(new_agent_jid)

                    new_device_agent = DeviceAgent(new_agent_jid, new_agent_password)
                    await new_device_agent.start(auto_register=True)
                except Exception as e:
                    fail_text = str(e)
                    new_device_agent = None

                if not new_device_agent or not new_device_agent.is_alive():
                    logger.warning(f'{self.agent.jid}: Failed to add new device agent {new_agent_jid}. {fail_text}')
                    error_msg = ErrorMessage(msg)
                    error_msg.body = f'Failed to add new device agent. {fail_text}'
                    await self.send(error_msg)
                    return
                
                self.get('device_agents').append(new_device_agent)

                reply_msg = SuccessMessage(msg)
                await self.send(reply_msg)
                logger.debug(f'{self.agent.jid}: New device agent {new_agent_jid} added')

    class ReceiveNewStateBehavior(CyclicBehaviour):
        '''Behavior to receive a message with a new state. 
        The state is then propagated to all device prediction agents.'''
        async def run(self):
            self.agent: MainAgent
            msg = await self.receive(timeout=10)
            if msg:
                msg = NewStateMessage.from_message(msg)
                logger.debug(f'{self.agent.jid}: New state received with content: {msg.State}')

                for device_agent in self.get('device_agents'):
                    new_state_message = NewStateMessage()
                    new_state_message.to = str(device_agent.jid)
                    new_state_message.body = msg.body
                    await self.send(new_state_message)

                if prediction_config.predict_on_new_state and not prediction_config.periodic_prediction[0]:
                    # NOTE: Collect user locations when predicting only on new state
                    for user, location in msg.State.UserLocations.items():
                        SystemStats().users_locations[user].append(location)

                # NOTE: Relies on agents' behaviors being 
                # processed right after sending them a message 
                # and then sending a reply after their behaviors are done
                if msg.metadata.get('reply-with'):
                    reply_msg = SuccessMessage(msg)
                    await self.send(reply_msg)

    class ReceiveTriggerPredictionBehavior(CyclicBehaviour):
        '''Behavior to receive a message to trigger predictions.
        The message is then sent to all device agents.'''
        async def run(self):
            self.agent: MainAgent
            msg = await self.receive(timeout=10)
            if msg:
                msg = TriggerPredictionMessage.from_message(msg)
                logger.debug(f'{self.agent.jid}: Trigger predictions received with content: {msg.body}')

                for device_agent in self.get('device_agents'):
                    trigger_predictions_message = TriggerPredictionMessage()
                    trigger_predictions_message.to = str(device_agent.jid)
                    await self.send(trigger_predictions_message)

                # NOTE: Relies on agents' behaviors being 
                # processed right after sending them a message 
                # and then sending a reply after their behaviors are done
                if msg.metadata.get('reply-with'):
                    reply_msg = SuccessMessage(msg)
                    await self.send(reply_msg)

    class ReceivePredictionBehavior(CyclicBehaviour):
        '''Behavior to receive a message with a prediction.
        The prediciton is then propagated to all user agents.'''
        async def run(self):
            self.agent: MainAgent
            msg = await self.receive(timeout=10)
            if msg:
                msg = PredictionMessage.from_message(msg)
                logger.debug(f'{self.agent.jid}: Prediction received with content: {msg.body}')

                for user_agent in self.get('user_agents'):
                    prediction_message = PredictionMessage()
                    prediction_message.to = str(user_agent.jid)
                    prediction_message.body = msg.body
                    await self.send(prediction_message)

    class ReceiveActionBehavior(CyclicBehaviour):
        '''Behavior to receive a message with action.
        The action is then sent to the system interface.'''
        async def run(self):
            self.agent: MainAgent
            msg = await self.receive(timeout=10)
            if msg:
                msg = ActionMessage.from_message(msg)
                logger.debug(f'{self.agent.jid}: Action received with content: {msg.body}')

                msg.to = interface_jid or ""
                msg.sender = str(self.agent.jid)
                await self.send(msg)

    async def setup(self):
        await super().setup()

        global main_agent_jid
        main_agent_jid = str(self.jid)

        self.set('user_agents', [])
        self.set('device_agents', [])

        receive_add_new_user_behavior = self.ReceiveAddNewUserAgentBehavior()
        add_new_user_message_template = AddNewUserAgentMessage()
        self.add_behaviour(receive_add_new_user_behavior, add_new_user_message_template)

        receive_add_new_device_agent_behavior = self.ReceiveAddNewDeviceAgentBehavior()
        add_new_device_agent_message_template = AddNewDeviceAgentMessage()
        self.add_behaviour(receive_add_new_device_agent_behavior, add_new_device_agent_message_template)

        receive_new_state_behavior = self.ReceiveNewStateBehavior()
        new_state_message_template = NewStateMessage()
        self.add_behaviour(receive_new_state_behavior, new_state_message_template)

        receive_trigger_predictions_behavior = self.ReceiveTriggerPredictionBehavior()
        trigger_predictions_message_template = TriggerPredictionMessage()
        self.add_behaviour(receive_trigger_predictions_behavior, trigger_predictions_message_template)

        receive_prediction_behavior = self.ReceivePredictionBehavior()
        prediction_message_template = PredictionMessage()
        self.add_behaviour(receive_prediction_behavior, prediction_message_template)

        receive_action_behavior = self.ReceiveActionBehavior()
        action_message_template = ActionMessage()
        self.add_behaviour(receive_action_behavior, action_message_template)

        send_ready_message_behavior = self.SendReadyMessageBehavior()
        self.add_behaviour(send_ready_message_behavior)

    async def stop(self):
        for agent in self.get('user_agents') + self.get('device_agents'):
            await agent.stop()
        try:
            SystemStats().save_plots(f'{prediction_config.models_folder}/plots.png')
        except:
            logger.error('Error saving plots')

        await super().stop()


class UserAgent(BaseAgent):
    '''Agent used to send device actions based on user settings and PredictionAgents' predictions.'''
    class RecievePredictionBehavior(CyclicBehaviour):
        async def run(self):
            self.agent: UserAgent
            msg = await self.receive(timeout=10)
            if msg:
                msg = PredictionMessage.from_message(msg)
                logger.debug(f'{self.agent.jid}: Prediction received with content: {msg.body}')

                device_jid, pred = msg.body.split(' ')
                pred = float(pred)

                # Filter prediction
                device_filters: List[DeviceFilter] = self.get('device_filters')
                if device_jid not in device_filters or not device_filters[device_jid].Enabled:
                    logger.debug(f'{self.agent.jid}: Device filter not set or disabled. Ignoring prediction.')
                    return
                device_filter: DeviceFilter = device_filters[device_jid]
                if pred <= (device_filter.Treshold_Off or 0.0):
                    pred = 0
                elif pred >= (device_filter.Treshold_On or 1.0):
                    pred = 1
                else:
                    logger.info(f'{self.agent.jid}: Prediction not within tresholds. Ignoring prediction.')
                    return

                time = get_time()

                action_msg = ActionMessage()
                action_msg.to = main_agent_jid or ""
                action_msg.Action = time, device_jid, pred
                await self.send(action_msg)
                logger.info(f'{self.agent.jid}: Action sent for device {device_jid} with desired state {pred} and at time {time} ')

    class ReceiveSetDeviceFilterBehavior(CyclicBehaviour):
        async def run(self):
            self.agent: UserAgent
            msg = await self.receive(timeout=10)
            if msg:
                msg = SetDeviceFilterMessage.from_message(msg)
                logger.debug(f'{self.agent.jid}: Set device filter received with content: {msg.body}')

                msg_device_filter: DeviceFilter = msg.FilterSettings

                device_filters = self.get('device_filters')
                if msg_device_filter.Device_JID not in device_filters:
                    # Add new device filter
                    new_device_filter = DeviceFilter(Device_JID=msg_device_filter.Device_JID, 
                                                     Enabled=msg_device_filter.Enabled or False,
                                                     Treshold_Off=msg_device_filter.Treshold_Off or 0.5,
                                                     Treshold_On=msg_device_filter.Treshold_On or 0.5)
                    device_filters[msg_device_filter.Device_JID]  = new_device_filter
                else:
                    device_filter: DeviceFilter = device_filters[msg_device_filter.Device_JID]
                    # Update device filter
                    if msg_device_filter.Enabled is not None:
                        device_filter.Enabled = msg_device_filter.Enabled
                    if msg_device_filter.Treshold_Off is not None:
                        device_filter.Treshold_Off = msg_device_filter.Treshold_Off
                    if msg_device_filter.Treshold_On is not None:
                        device_filter.Treshold_On = msg_device_filter.Treshold_On

                reply_msg = SuccessMessage(msg)
                await self.send(reply_msg)

                logger.info(f'{self.agent.jid}: Device filter set for {msg_device_filter.Device_JID}. {self.get("device_filters")[msg_device_filter.Device_JID]}')

    async def setup(self):
        await super().setup()

        receive_prediction_behavior = self.RecievePredictionBehavior()
        prediction_message_template = PredictionMessage()
        self.add_behaviour(receive_prediction_behavior, prediction_message_template)

        self.set('device_filters', {})
        receive_set_device_filter_behavior = self.ReceiveSetDeviceFilterBehavior()
        set_device_filter_message_template = SetDeviceFilterMessage()
        self.add_behaviour(receive_set_device_filter_behavior, set_device_filter_message_template)


class DeviceAgent(BaseAgent):
    '''Agent used to receive state updates and send predictions to UserAgents.'''
    class ReceiveNewStateBehavior(CyclicBehaviour):
        async def run(self):
            self.agent: DeviceAgent
            msg = await self.receive(timeout=10)
            if msg:
                msg = NewStateMessage.from_message(msg)
                logger.debug(f'{self.agent.jid}: New state received with content: {msg.State}')
                
                # user locations, this device's state
                state = list(msg.State.UserLocations.values())
                if str(self.agent.jid) not in msg.State.DeviceStates:
                    logger.info(f'{self.agent.jid}: No state for this device in state update. Using last state.')
                    return
                state += [msg.State.DeviceStates[str(self.agent.jid)]]
                
                if self.get('model_initialized') and len(state) != self.get('state_size'):
                    logger.warning(f'{self.agent.jid}: Number of state features doesn\'t match the model input size. Using last state.')
                    return
                    
                self.set('state', state)

                if not self.get('model_initialized'):
                    if isinstance(prediction_config, PredictionConfig):
                        self.set('model_initialized', True)
                        self.set('state_size', len(state))
                        # Initialize model with time size + state size as input size
                        input_size = len(TimeSlot(0,0,0)) + len(state)
                        model = PredictionModel(input_size=input_size,
                                                model_params=prediction_config.model_params, 
                                                model_path=os.path.join(prediction_config.models_folder,f'{self.agent.jid}.pth'), 
                                                save_after_n_learning_steps=prediction_config.save_after_n_learning_steps,
                                                collect_stats_for_device=str(self.agent.jid))
                        if prediction_config.load_model:
                            model.load()
                        self.set("model", model)
                        # Periodically send predictions 
                        if prediction_config.periodic_prediction[0]:
                            periodic_prediction_behavior = self.agent.PeriodicPredictionsBehavior(prediction_config.periodic_prediction[1]*60)
                            self.agent.add_behaviour(periodic_prediction_behavior)
                    else:
                        logger.error(f'{self.agent.jid}: Prediction configuration not set. Prediction not enabled.')
                
                # Predict on new state
                if prediction_config.predict_on_new_state and self.get('model_initialized'):
                    self.agent.add_behaviour(self.agent.PredictBehavior())

    class ReceiveTriggerPredictionBehavior(CyclicBehaviour):
        async def run(self):
            self.agent: DeviceAgent
            msg = await self.receive(timeout=10)
            if msg:
                msg = TriggerPredictionMessage.from_message(msg)
                logger.debug(f'{self.agent.jid}: Trigger predictions received with content: {msg.body}')
                self.agent.add_behaviour(self.agent.PredictBehavior())

    class PeriodicPredictionsBehavior(PeriodicBehaviour):
        async def run(self):
            self.agent: DeviceAgent
            self.agent.add_behaviour(self.agent.PredictBehavior())

        async def on_start(self):
            logger.info(f'{self.agent.jid}: Periodic prediction behavior started')

    class PredictBehavior(OneShotBehaviour):
        async def run(self):
            self.agent: DeviceAgent
            state = self.get('state')
            time = get_time()
            state = [time.Hour, time.Minute, time.DayOfWeek, *state]

            model: PredictionModel|None = self.agent.get('model')
            if not model:
                logger.warning(f'{self.agent.jid}: Model not initialized. Prediction not sent.')
                return

            # Learn from last prediction and current device state
            if self.get('learning'):
                model.learn(state[-1])
            else:
                # Current prediction will be used next time
                self.set('learning', True)

            # Make prediction
            prediction = model.predict(state)
            logger.info(f'{self.agent.jid}: State: {state} Prediction: {prediction}')
            
            msg = PredictionMessage()
            msg.to = main_agent_jid or ""
            msg.Prediction = str(self.agent.jid), prediction
            await self.send(msg)
            logger.debug(f'{self.agent.jid}: Prediction sent')

    async def setup(self):
        await super().setup()

        new_state_behavior = self.ReceiveNewStateBehavior()
        new_state_message_template = NewStateMessage()
        self.add_behaviour(new_state_behavior, new_state_message_template)

        trigger_prediction_behavior = self.ReceiveTriggerPredictionBehavior()
        trigger_prediction_message_template = TriggerPredictionMessage()
        self.add_behaviour(trigger_prediction_behavior, trigger_prediction_message_template)

        self.set('model_initialized', False)
        self.set('learning', False)
        # NOTE: Prediction behavior is added when the first state is received

    async def stop(self):
        if self.is_alive():
            model: PredictionModel|None = self.get('model')
            if model and prediction_config.save_model:
                saved = model.save()
                if not saved:
                    logger.error(f'{self.jid}: Model not saved')
        await super().stop()