import pickle
import socket

from MAS.messages.spadeMessages import *
from systemTestTCP import SOCKET_HOST, SOCKET_PORT


def send_message(msg: Message, sckt: socket.socket):
    serialized_msg = pickle.dumps(msg)
    msg_length = len(serialized_msg)
    sckt.sendall(msg_length.to_bytes(4, byteorder='big'))
    sckt.sendall(serialized_msg)

# Connect
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((SOCKET_HOST, SOCKET_PORT))

if input('# Set filter? (y/n) \n') == 'y':
    # Set filter
    msg = SetDeviceFilterMessage()
    device_filter = DeviceFilter('deviceagent@localhost', True, 0.5, 0.5)
    msg.body = device_filter.to_json()
    msg.to = 'useragent@localhost'
    send_message(msg, s)

# Send state
while input('# Send state update? (y/n) \n') == 'y':
    msg = NewStateMessage()
    msg.State = State({'useragent@localhost': 1}, {'deviceagent@localhost': 1})
    msg.to = 'mainagent@localhost'
    send_message(msg, s)
    print(f'State sent')
    
# Receive a message
# data = s.recv(1024)
# print(f'Received: {data}')

# Send stop message
msg = StopMessage()
msg.to = 'mainagent@localhost'
send_message(msg, s)
print(f'Stop message sent')

s.close()