from spade.message import Message, MessageBase
from spade.template import Template
from typing import Tuple, Optional

from MAS.data import *

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
        self.to = str(reply_to.sender)
        
    @classmethod
    def from_message(cls, msg: Message) -> 'ReplyMessage':
        """Create a ReplyMessage from a received Message"""
        message = cls(msg)
        message.sender = str(msg.sender)
        message.to = str(msg.to)
        message.body = msg.body
        for key, value in msg.metadata.items():
            message.set_metadata(key, value)
        return message

class SuccessMessage(Message):
    def __init__(self, reply_to: Message):
        super().__init__()
        self.set_metadata("performative", "inform")
        self.set_metadata("ontology", "success")
        if reply_with := reply_to.metadata.get("reply-with"):
            self.set_metadata("in-reply-to", reply_with)
        self.to = str(reply_to.sender)
        self.body = reply_to.body

    @classmethod
    def from_message(cls, msg: Message) -> 'SuccessMessage':
        """Create a SuccessMessage from a received Message"""
        original_msg = Message()  # Create a dummy message to reply to
        original_msg.sender = str(msg.to)
        original_msg.set_metadata("reply-with", msg.metadata.get("in-reply-to") or "")
        
        message = cls(original_msg)
        message.sender = str(msg.sender)
        message.to = str(msg.to)
        message.body = msg.body
        for key, value in msg.metadata.items():
            message.set_metadata(key, value)
        return message

class ErrorMessage(Message):
    def __init__(self, reply_to: Optional[Message] = None):
        super().__init__()
        self.set_metadata("performative", "failure")
        self.set_metadata("ontology", "error")
        if reply_to:
            if reply_with := reply_to.metadata.get("reply-with"):
                self.set_metadata("in-reply-to", reply_with)
            self.to = str(reply_to.sender)
            
    @classmethod
    def from_message(cls, msg: Message) -> 'ErrorMessage':
        """Create an ErrorMessage from a received Message"""
        message = cls()
        message.sender = str(msg.sender)
        message.to = str(msg.to)
        message.body = msg.body
        for key, value in msg.metadata.items():
            message.set_metadata(key, value)
        return message

###
###
###

class AgentReadyMessage(Message):
    def __init__(self):
        super().__init__()
        self.set_metadata("performative", "inform")
        self.set_metadata("ontology", "agentReady")
        
    @classmethod
    def from_message(cls, msg: Message) -> 'AgentReadyMessage':
        """Create an AgentReadyMessage from a received Message"""
        message = cls()
        message.sender = str(msg.sender)
        message.to = str(msg.to)
        message.body = msg.body
        for key, value in msg.metadata.items():
            message.set_metadata(key, value)
        return message

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
    def State(self, state: "State"):
        self.body = state.to_json()
    
    def match(self, msg: type[MessageBase]):
        return bool(super().match(msg))
        
    @classmethod
    def from_message(cls, msg: Message) -> 'NewStateMessage':
        """Create a NewStateMessage from a received Message"""
        message = cls()
        message.sender = str(msg.sender)
        message.to = str(msg.to)
        message.body = msg.body
        for key, value in msg.metadata.items():
            message.set_metadata(key, value)
        return message

class StopMessage(Message):
    def __init__(self):
        super().__init__()
        self.set_metadata("performative", "request")
        self.set_metadata("ontology", "stop")
        
    @classmethod
    def from_message(cls, msg: Message) -> 'StopMessage':
        """Create a StopMessage from a received Message"""
        message = cls()
        message.sender = str(msg.sender)
        message.to = str(msg.to)
        message.body = msg.body
        for key, value in msg.metadata.items():
            message.set_metadata(key, value)
        return message

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
        return (device_jid, float(desired_state))
    
    @Prediction.setter
    def Prediction(self, prediction: Tuple[str, int|float]):
        self.body = f'{prediction[0]} {prediction[1]}'

    def match(self, msg: MessageBase) -> bool:
        if not super().match(msg): # type: ignore
            return False

        try:
            device_jid, desired_state = msg.body.split(' ')
            if not device_jid or not desired_state:
                raise ValueError()
            float(desired_state)
        except Exception:
            return False

        return True

    @classmethod
    def from_message(cls, msg: Message) -> 'PredictionMessage':
        """Create a PredictionMessage from a received Message"""
        message = cls()
        message.sender = str(msg.sender)
        message.to = str(msg.to)
        message.body = msg.body
        for key, value in msg.metadata.items():
            message.set_metadata(key, value)
        return message

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
        timeslot_Minute_str, timeslot_Hour_str, timeslot_DayOfWeek_str, device_jid, desired_state_str = self.body.split(' ')
        timeslot.Minute = int(timeslot_Minute_str)
        timeslot.Hour = int(timeslot_Hour_str)
        timeslot.DayOfWeek = int(timeslot_DayOfWeek_str)
        desired_state = float(desired_state_str)
        return (timeslot, device_jid, desired_state)
    
    @Action.setter
    def Action(self, action: Tuple[TimeSlot, str, int|float]):
        self.body = f'{action[0].Minute} {action[0].Hour} {action[0].DayOfWeek} {action[1]} {action[2]}'
        
    @classmethod
    def from_message(cls, msg: Message) -> 'ActionMessage':
        """Create an ActionMessage from a received Message"""
        message = cls()
        message.sender = str(msg.sender)
        message.to = str(msg.to)
        message.body = msg.body
        for key, value in msg.metadata.items():
            message.set_metadata(key, value)
        return message

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
            self.body = ' password'

    def match(self, msg: MessageBase):
        if not super().match(msg): # type: ignore
            return False

        # Check if the message body contains 'JID password'
        if not msg.body:
            return False
        jid, password = msg.body.split(' ')
        return bool(jid and password)
        
    @classmethod
    def from_message(cls, msg: Message) -> 'AddNewUserAgentMessage':
        """Create an AddNewUserAgentMessage from a received Message"""
        message = cls()
        message.sender = str(msg.sender)
        message.to = str(msg.to)
        message.body = msg.body
        for key, value in msg.metadata.items():
            message.set_metadata(key, value)
        return message

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
        if not super().match(msg): # type: ignore
            return False

        try:
            DeviceFilter.from_json(msg.body)
        except Exception:
            return False

        return True
        
    @classmethod
    def from_message(cls, msg: Message) -> 'SetDeviceFilterMessage':
        """Create a SetDeviceFilterMessage from a received Message"""
        message = cls()
        message.sender = str(msg.sender)
        message.to = str(msg.to)
        message.body = msg.body
        for key, value in msg.metadata.items():
            message.set_metadata(key, value)
        return message

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
            self.body = ' password'

    def match(self, msg: MessageBase):
        if not super().match(msg): # type: ignore
            return False

        # Check if the message body contains 'JID password'
        if not msg.body:
            return False
        jid, password = msg.body.split(' ')
        return bool(jid and password)
        
    @classmethod
    def from_message(cls, msg: Message) -> 'AddNewDeviceAgentMessage':
        """Create an AddNewDeviceAgentMessage from a received Message"""
        message = cls()
        message.sender = str(msg.sender)
        message.to = str(msg.to)
        message.body = msg.body
        for key, value in msg.metadata.items():
            message.set_metadata(key, value)
        return message
    
class TriggerPredictionMessage(Message):
    def __init__(self):
        super().__init__()
        self.set_metadata("performative", "request")
        self.set_metadata("ontology", "triggerPredictions")
        
    @classmethod
    def from_message(cls, msg: Message) -> 'TriggerPredictionMessage':
        """Create a TriggerPredictionMessage from a received Message"""
        message = cls()
        message.sender = str(msg.sender)
        message.to = str(msg.to)
        message.body = msg.body
        for key, value in msg.metadata.items():
            message.set_metadata(key, value)
        return message