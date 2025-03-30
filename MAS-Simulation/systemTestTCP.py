import pickle
import socket
import select
import logging
import time

import spade
import asyncio
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour, OneShotBehaviour
from MAS.messages.spadeMessages import *
from MAS.data import *
from MAS import agents
from MAS.system import default_get_time_func, configure_logging

SOCKET_HOST = 'localhost'
SOCKET_PORT = 50007

logger = logging.getLogger('MAS.systemTest')

class TestAgent(agents.BaseAgent):

    # Measuring time for state update
    class SendStateBehavior(PeriodicBehaviour):
        async def run(self):
            msg = NewStateMessage()
            msg.State = State({'useragent@localhost': 1}, {'deviceagent@localhost': 1})
            msg.set_metadata('reply-with', 'state')
            msg.to = agents.main_agent_jid

            start_time = time.time()
            await self.send(msg)
            logger.info(f'State sent in {time.time() - start_time} seconds')

            reply = await self.receive(timeout=5)
            logger.info(f'{self.agent.jid} Reply received in {time.time() - start_time} seconds')
            if reply:
                logger.info(f'{self.agent.jid} Received reply: {reply}')
            else:
                logger.info(f'{self.agent.jid} No reply')

    # Create inital agents
    class SpawnAgentsBehavior(OneShotBehaviour): # TODO: Try with Periodic (see how much longer does state update take)
        async def run(self):
            new_user_msg = AddNewUserAgentMessage()
            new_user_msg.set_metadata('reply-with', 'new_user')
            new_user_msg.body = 'useragent@localhost password'
            new_user_msg.to = agents.main_agent_jid
            await self.send(new_user_msg)
            await self.receive(timeout=5)
            logger.info(f'{self.agent.jid} New user agent spawned')


            new_device_msg = AddNewDeviceAgentMessage()
            new_device_msg.set_metadata('reply-with', 'new_device')
            new_device_msg.body = 'deviceagent@localhost password'
            new_device_msg.to = agents.main_agent_jid
            await self.send(new_device_msg)
            await self.receive(timeout=5)
            logger.info(f'{self.agent.jid} New device agent spawned')

            # self.agent.add_behaviour(self.agent.SendStateBehavior(period=5))

    # Relay messages from socket
    class RelayMessagesFromSocketBehavior(CyclicBehaviour):
        async def run(self):
            conn: socket.socket = self.agent.get('conn')
            if not conn:
                logger.error(f'{self.agent.jid} Missing socket')
                self.kill()
                return
            
            poller_events, _, _ = select.select([conn], [], [], 0)
            if poller_events:
                logger.info(f'{self.agent.jid} Poller events: {poller_events}')
                
                msg = None
                try:
                    msg_size = conn.recv(4)
                    print(f'Message size: {msg_size}')
                    data = conn.recv(int.from_bytes(msg_size, byteorder='big'))
                    # data = b''
                    # while True:
                    #     chunk = conn.recv(4096)
                    #     if not chunk:
                    #         break
                    #     data += chunk
                    msg: Message = pickle.loads(data)
                    logger.info(f'{self.agent.jid} Message received from client: {msg}')
                except Exception as e:
                    logger.error(f'{self.agent.jid} Error processing message: {e}')

                if msg:
                    msg.sender = str(self.agent.jid)
                    await self.send(msg)

                    if StopMessage().match(msg):
                        await self.agent.stop()
            
            await asyncio.sleep(0.1)


    async def setup(self):
        await super().setup()

        # Open socket for communication
        sockt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sockt.bind((SOCKET_HOST, SOCKET_PORT))
        sockt.listen(1)
        self.set('socket', sockt)

        # Wait for connection
        logger.info(f'{self.jid} Listening on {SOCKET_HOST}:{SOCKET_PORT}')
        conn, addr = sockt.accept()
        logger.info(f'{self.jid} Connected by {addr}')
        self.set('conn', conn)

        self.add_behaviour(self.SpawnAgentsBehavior())
        self.add_behaviour(self.RelayMessagesFromSocketBehavior())

    async def stop(self):
        await super().stop()
        conn: socket.socket = self.get('conn')
        if conn:
            conn.close()
            self.set('conn', None)
        sockt: socket.socket = self.get('socket')
        if sockt:
            sockt.close()
            self.set('socket', None)

    def __del__(self):
        conn: socket.socket = self.get('conn')
        if conn:
            conn.close()
        sockt: socket.socket = self.get('socket')
        if sockt:
            sockt.close()
        super().__del__()


async def test_main(main_agent_jid: str, main_agent_password: str):
    mainAgent = agents.MainAgent(main_agent_jid, main_agent_password)
    await mainAgent.start()

    testAgent = TestAgent('testagent@localhost', 'password')
    await testAgent.start()

    await spade.wait_until_finished([mainAgent, testAgent])


if __name__ == '__main__':
    configure_logging(log_conf=logging.DEBUG)
    agents.get_time = default_get_time_func
    agents.interface_jid = 'testagent@localhost'
    spade.run(test_main('mainagent@localhost', 'password'))
    logger.info('All agents finished')