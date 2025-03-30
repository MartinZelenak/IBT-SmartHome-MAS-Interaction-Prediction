from typing import Dict, List, Optional, Tuple

import xmpp
from MAS.data import *

# REVIEW: Unify body parsing/validation being done in one place (match() method or interface)

# Example of a message with SPADE metadata:
# <message xml:lang='en'
#         to='interface@localhost'
#         from='agent@localhost/5646539464458296395994'
#         type='chat'
#         id=':RbJv-Jg5RIYLeUP8tQqK'
#         xmlns='jabber:client'>
#  <archived by='interface@localhost'
#         id='1710017948006038'
#         xmlns='urn:xmpp:mam:tmp'/>
#  <stanza-id by='interface@localhost'
#             id='1710017948006038'
#             xmlns='urn:xmpp:sid:0'/>
#  <x xmlns='jabber:x:data'
#     type='form'>
#   <field var='performative'
#           type='text-single'>
#       <value>inform</value>
#   </field>
#   <field var='ontology'
#           type='text-single'>
#       <value>MyOntology</value>
#   </field>
#   <title>spade:x:metadata</title>
#  </x>
#  <body>Message text</body>
# </message>

def get_spade_metadata_from_xmpp_message(message: xmpp.Message) -> Dict[str, str]|None:
    '''Get the spade metadata from an xmpp.Message.'''
    x: xmpp.Node|None = message.getTag(name='x')
    if not x:
        return None
    metadata = {}
    children: List[xmpp.Node] = x.getChildren()
    for child in children:
        if child.getName() != 'field':
            continue
        key = child.getAttr('var')
        value = child.getTagData('value')
        metadata[key] = value
    return metadata if metadata else None

class Message(xmpp.Message):
    '''A XMPP message with SPADE metadata.'''
    def __init__(self, body: Optional[str] = None, to: Optional[str] = None):
        super().__init__(body=body, to=to)
        self.setAttr('type', 'chat')
        self.has_metadata = False
        self.metadata = {}

    def set_spade_metadata(self, key: str, value: str):
        '''Set the spade metadata for the message.'''
        if not self.has_metadata:
            x = self.addChild(name='x', namespace='jabber:x:data', attrs={'type': 'form'})
            x.addChild(name='title').setData('spade:x:metadata')
            self.has_metadata = True
        else:
            x = self.getTag(name='x')

        field = x.getTag(name='field', attrs={'var': key})
        if not field:
            # Add new field
            field = x.addChild(name='field', attrs={'var': key, 'type': 'text-single'})
            field.addChild(name='value').setData(value)
        else:
            # Update existing field
            field.getTag(name='value').setData(value)

    def get_spade_metadata(self, key: Optional[str] = None) -> Optional[Dict[str, str]|str]:
        '''Get the spade metadata for the message.
        If key is None, returns all metadata.
        If message has no metadata or key is not found, returns None.
        If key is found, returns the value of the key.'''
        x: xmpp.Node|None = self.getTag(name='x')
        if not x:
            return None
        if not key:
            metadata = {}
            children: List[xmpp.Node] = x.getChildren()
            for child in children:
                if child.getName() != 'field':
                    continue
                key = child.getAttr('var')
                value = child.getTagData('value')
                metadata[key] = value
            return metadata if metadata else None
        
        field: xmpp.Node|None = x.getTag(name='field', attrs={'var': key})
        if not field:
            return None
        return field.getTagData('value')
    
    def expect_reply(self, reply_with: str):
        '''Set the reply-with metadata for the message.'''
        self.set_spade_metadata('reply-with', reply_with)

    def match(self, message: xmpp.Message) -> bool:
        '''Match the message with another message.'''
        for attr in ['to', 'from', 'type', 'body']:
            if self.getAttr(attr) and self.getAttr(attr) != message.getAttr(attr):
                return False
            
        message_metadata = get_spade_metadata_from_xmpp_message(message)
        self_metadata = self.get_spade_metadata()
        if message_metadata and self_metadata:
            for key, value in self_metadata.items():
                if key not in message_metadata or message_metadata[key] != value:
                    return False
        elif message_metadata or self_metadata:
            return False

        return True

################################
#### ONLY OUTGOING MESSAGES ####
################################
class StopMessage(Message):
    def __init__(self, to: Optional[str] = None):
        super().__init__(body='Stop', to=to)
        self.set_spade_metadata('performative', 'request')
        self.set_spade_metadata('ontology', 'stop')

class NewStateMessage(Message):
    def __init__(self, state: State, to: Optional[str] = None):
        super().__init__(body=state.to_json(), to=to)
        self.set_spade_metadata('performative', 'inform')
        self.set_spade_metadata('language', 'json')
        self.set_spade_metadata('ontology', 'newState')

    @property
    def State(self) -> State:
        return State.from_json(self.body)
    @State.setter
    def State(self, state: State):
        self.body = state.to_json()

class AddNewUserAgentMessage(Message):
    def __init__(self, jid: str, password: str, to: Optional[str] = None):
        super().__init__(body=f'{jid} {password}', to=to)
        self.set_spade_metadata('performative', 'request')
        self.set_spade_metadata('ontology', 'newUser')
        self.set_spade_metadata('reply-with', f'add-user-{str(self.getID())}')

class SetDeviceFilterMessage(Message):
    def __init__(self, filter_settings: DeviceFilter, to: Optional[str] = None):
        super().__init__(body=filter_settings.to_json(), to=to)
        self.set_spade_metadata('performative', 'request')
        self.set_spade_metadata('ontology', 'setDeviceFilter')
        self.set_spade_metadata('language', 'json')
        self.set_spade_metadata('reply-with', f'set-filter-{str(self.getID())}')

    @property
    def FilterSettings(self) -> DeviceFilter:
        return DeviceFilter.from_json(self.body)
    @FilterSettings.setter
    def FilterSettings(self, filter_settings: DeviceFilter):
        self.body = filter_settings.to_json()

class AddNewDeviceAgentMessage(Message):
    def __init__(self, jid: str, password: str, to: Optional[str] = None):
        super().__init__(body=f'{jid} {password}', to=to)
        self.set_spade_metadata('performative', 'request')
        self.set_spade_metadata('ontology', 'newDevice')
        self.set_spade_metadata('reply-with', f'add-device-{str(self.getID())}')

class TriggerPredictionMessage(Message):
    def __init__(self, to: Optional[str] = None):
        super().__init__(body=None, to=to)
        self.set_spade_metadata('performative', 'request')
        self.set_spade_metadata('ontology', 'triggerPredictions')


#################################
#### ONLY INCOMMING MESSAGES ####
#################################
class SuccessMessage(Message):
    def __init__(self, to: Optional[str] = None):
        super().__init__(body=None, to=to)
        self.set_spade_metadata('performative', 'inform')
        self.set_spade_metadata('ontology', 'success')

class ErrorMessage(Message):
    def __init__(self, to: Optional[str] = None):
        super().__init__(body=None, to=to)
        self.set_spade_metadata('performative', 'failure')
        self.set_spade_metadata('ontology', 'error')

class AgentReadyMessage(Message):
    def __init__(self, to: Optional[str] = None):
        super().__init__(body=None, to=to)
        self.set_spade_metadata('performative', 'inform')
        self.set_spade_metadata('ontology', 'agentReady')

class ActionMessage(Message):
    def __init__(self, action: Optional[str|Tuple[TimeSlot, str, int|float]] = None, to: Optional[str] = None):
        super().__init__(body=None, to=to)
        self.set_spade_metadata('performative', 'request')
        self.set_spade_metadata('ontology', 'takeAction')
        if action:
            if isinstance(action, str):
                self.setBody(action)
            else:
                self.Action = action

    @property
    def Action(self) -> Tuple[TimeSlot, str, int|float]|None:
        body: str|None =  self.getBody()
        if body is None:
            return None
        timeslot = TimeSlot(0,0,0)
        timeslot.Minute, timeslot.Hour, timeslot.DayOfWeek, device_jid, desired_state = body.split(' ')
        
        # Check if desired_state string is an Int or Float
        try:
            desired_state = int(desired_state)
        except ValueError:
            desired_state = float(desired_state)

        return (timeslot, device_jid, float(desired_state))
    
    @Action.setter
    def Action(self, action: Tuple[TimeSlot, str, int|float]):
        self.setBody(f'{action[0].Minute} {action[0].Hour} {action[0].DayOfWeek} {action[1]} {action[2]}')

    def match(self, message: xmpp.Message) -> bool:
        if not super().match(message):
            return False
        
        try:
            timeslot = TimeSlot(0,0,0)
            timeslot.Minute, timeslot.Hour, timeslot.DayOfWeek, device_jid, desired_state = message.getBody().split(' ')
            if not device_jid or not desired_state:
                raise ValueError()
            float(desired_state)
        except:
            return False
        
        return True