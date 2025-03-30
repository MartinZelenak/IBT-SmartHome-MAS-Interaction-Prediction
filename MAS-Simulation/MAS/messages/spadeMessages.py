from spade.message import Message, MessageBase
from spade.template import Template
from typing import Tuple, Optional

from MAS.data import *

# REVIEW: Unify body parsing/validation being done in one place (match() method or agent)

### 
### REPLY MESSAGES
### 
class ReplyMessage(Message):
    def __init__(self, reply_to: Message):
        super().__init__()
        self.set_metadata("performative", "inform")
        self.set_metadata("ontology", "reply")
        if reply_to.metadata.get("reply-with"):
            self.set_metadata("in-reply-to", reply_to.metadata["in-reply-to"])
        self.to = reply_to.sender

class SuccessMessage(Message):
    def __init__(self, reply_to: Message):
        super().__init__()
        self.set_metadata("performative", "inform")
        self.set_metadata("ontology", "success")
        reply_with = reply_to.metadata.get("reply-with")
        if reply_with:
            self.set_metadata("in-reply-to", reply_with)
        self.to = str(reply_to.sender)
        self.body = reply_to.body

class ErrorMessage(Message):
    def __init__(self, reply_to: Optional[Message] = None):
        super().__init__()
        self.set_metadata("performative", "failure")
        self.set_metadata("ontology", "error")
        if reply_to:
            reply_with = reply_to.metadata.get("reply-with")
            if reply_with:
                self.set_metadata("in-reply-to", reply_with)
            self.to = str(reply_to.sender)

###
###
###

class AgentReadyMessage(Message):
    def __init__(self):
        super().__init__()
        self.set_metadata("performative", "inform")
        self.set_metadata("ontology", "agentReady")

class NewStateMessage(Message):
    def __init__(self):
        super().__init__()
        self.set_metadata("performative", "inform")
        self.set_metadata("language", "json")
        self.set_metadata("ontology", "newState")

    @property
    def State(self) -> State:
        return State.from_json(self.body)
    @State.setter
    def State(self, state: State):
        self.body = state.to_json()
    
    def match(self, msg: MessageBase):
        if not super().match(msg):
            return False
        # TODO: Other logic to match the message
        return True

class StopMessage(Message):
    def __init__(self):
        super().__init__()
        self.set_metadata("performative", "request")
        self.set_metadata("ontology", "stop")

class PredictionMessage(Message):
    def __init__(self):
        super().__init__()
        self.set_metadata("performative", "inform")
        self.set_metadata("ontology", "prediction")

    @property
    def Prediction(self) -> Tuple[str, int|float]|None:
        if not self.body:
            return None
        device_jid, desired_state = self.body.split(' ')
        return (device_jid, desired_state)
    
    @Prediction.setter
    def Prediction(self, prediction: Tuple[str, int|float]):
        self.body = f'{prediction[0]} {prediction[1]}'

    def match(self, message: MessageBase) -> bool:
        if not super().match(message):
            return False
        
        try:
            device_jid, desired_state = message.body.split(' ')
            if not device_jid or not desired_state:
                raise ValueError()
            float(desired_state)
        except:
            return False

        return True

class ActionMessage(Message):
    def __init__(self):
        super().__init__()
        self.set_metadata("performative", "request")
        self.set_metadata("ontology", "takeAction")
    
    @property
    def Action(self) -> Tuple[TimeSlot, str, int|float]|None:
        if not self.body:
            return None
        timeslot = TimeSlot(0,0,0)
        timeslot.Minute, timeslot.Hour, timeslot.DayOfWeek, device_jid, desired_state = self.body.split(' ')
        return (timeslot, device_jid, desired_state)
    
    @Action.setter
    def Action(self, action: Tuple[TimeSlot, str, int|float]):
        self.body = f'{action[0].Minute} {action[0].Hour} {action[0].DayOfWeek} {action[1]} {action[2]}'

class AddNewUserAgentMessage(Message):
    def __init__(self):
        super().__init__()
        self.set_metadata("performative", "request")
        self.set_metadata("ontology", "newUser")

    @property
    def JID(self) -> str:
        if self.body:
            jid, password = self.body.split(' ')
            return jid
        return ''
    @JID.setter
    def JID(self, jid: str):
        if self.body:
            password = self.body.split(' ')[1]
            self.body = f'{jid} {password}'
        else:
            self.body = jid

    @property
    def Password(self) -> str:
        if self.body:
            jid, password = self.body.split(' ')
            return password
        return ''
    @Password.setter
    def Password(self, password: str):
        if self.body:
            jid = self.body.split(' ')[0]
            self.body = f'{jid} {password}'
        else:
            self.body = f' password'

    def match(self, msg: MessageBase):
        if not super().match(msg):
            return False
        
        # Check if the message body contains 'JID password'
        if not msg.body:
            return False
        jid, password = msg.body.split(' ')
        if not jid or not password:
            return False

        return True

class SetDeviceFilterMessage(Message):
    def __init__(self):
        super().__init__()
        self.set_metadata("performative", "request")
        self.set_metadata("ontology", "setDeviceFilter")
        self.set_metadata("language", "json")

    @property
    def FilterSettings(self) -> DeviceFilter:
        return DeviceFilter.from_json(self.body)
    @FilterSettings.setter
    def FilterSettings(self, filter_settings: DeviceFilter):
        self.body = filter_settings.to_json()

    def match(self, msg: MessageBase):
        if not super().match(msg):
            return False
        
        try:
            DeviceFilter.from_json(msg.body)
        except:
            return False

        return True


class AddNewDeviceAgentMessage(Message):
    def __init__(self):
        super().__init__()
        self.set_metadata("performative", "request")
        self.set_metadata("ontology", "newDevice")

    @property
    def JID(self) -> str:
        if self.body:
            jid, password = self.body.split(' ')
            return jid
        return ''
    @JID.setter
    def JID(self, jid: str):
        if self.body:
            password = self.body.split(' ')[1]
            self.body = f'{jid} {password}'
        else:
            self.body = jid

    @property
    def Password(self) -> str:
        if self.body:
            jid, password = self.body.split(' ')
            return password
        return ''
    @Password.setter
    def Password(self, password: str):
        if self.body:
            jid = self.body.split(' ')[0]
            self.body = f'{jid} {password}'
        else:
            self.body = f' password'

    def match(self, msg: MessageBase):
        if not super().match(msg):
            return False
        
        # Check if the message body contains 'JID password'
        if not msg.body:
            return False
        jid, password = msg.body.split(' ')
        if not jid or not password:
            return False

        return True
    
class TriggerPredictionMessage(Message):
    def __init__(self):
        super().__init__()
        self.set_metadata("performative", "request")
        self.set_metadata("ontology", "triggerPredictions")