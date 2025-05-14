"""
Author: Martin Zelenák (xzelen27@stud.fit.vutbr.cz)
Description: Provides system startup, configuration, and main loop logic for the multi-agent system
            including agent initialization and logging setup.
Date: 2025-05-14
"""


import spade
import logging
import logging.config
import asyncio
import atexit
from typing import Callable, Optional, Iterable, Tuple, Any

from . import agents
from .systemStats import SystemStats
from .config import PredictionConfig
from .data import TimeSlot

logger = logging.getLogger('MAS.system')

def default_get_time_func() -> TimeSlot:
    import datetime
    now = datetime.datetime.now()
    return TimeSlot(now.minute, now.hour, now.weekday())

def configure_logging(log_conf: dict|int = logging.INFO):
    logging_set = False
    if isinstance(log_conf, dict):
        try:
            logging.config.dictConfig(log_conf)
            logging_set = True
        except Exception as e:
            logger.error(f'Failed to set logging configuration (using default): {e}')
            logging_set = False
    if not logging_set:
        logging.config.dictConfig({
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(name)s - %(levelname)s: %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default"
                }
            },
            "loggers": {
                "MAS": {
                    "level": log_conf if isinstance(log_conf, int) else logging.INFO,
                    "handlers": ["console"]
                },
                "spade": {
                    "level": logging.ERROR,
                    "handlers": ["console"]
                }
            }
        })

async def main(main_agent_jid: str, main_agent_password: str):
    main_agent = agents.MainAgent(main_agent_jid, main_agent_password)
    await main_agent.start()

    try:
        await spade.wait_until_finished(main_agent)
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt received. Stopping system...")

async def tcp_main(main_agent_jid: str, 
               main_agent_password: str, 
               tcp_interface_jid: str,
               tcp_interface_password: str,
               tcp_interface_host_ip: str = 'localhost',
               tcp_interface_port: int = 5007):
    main_agent = agents.MainAgent(main_agent_jid, main_agent_password)
    await main_agent.start()

    tcp_agent = agents.TCPRelayAgent(tcp_interface_jid, tcp_interface_password, tcp_interface_host_ip, tcp_interface_port)
    await tcp_agent.start()

    try:
        await spade.wait_until_finished(tcp_agent)
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt received. Stopping system...")
    
    
def system_start(interface_jid: str, 
                 main_agent_jid: str, 
                 main_agent_password: str, 
                 log_conf: Optional[dict|int] = logging.INFO,
                 prediction_conf: Optional[PredictionConfig] = None,
                 get_time_func: Optional[Callable[[], TimeSlot]] = None, 
                 get_time_func_params: Optional[Iterable[Any]] = None):
    '''Start the MAS system with the given configurations. This call blocks until the system is finished.
    To communicate with the system send messages to the given JID using XMPP.
    This is the standard way to start the system.

    Args:
        interface_jid (str): JID to send outgoing messages to
        main_agent_jid (str): JID of the main agent
        main_agent_password (str): password of the main agent
        log_conf (Optional[dict | int], optional): Logging configuration used in logging module. Defaults to logging.INFO.
        prediction_conf (Optional[PredictionConfig], optional): Configuration of the prediction model. Defaults to None.
        get_time_func (Optional[Callable[[], TimeSlot]], optional): Function for getting the current environment time. If None, defaults to using datetime.
        get_time_func_params (Iterable[Any], optional): Prameters to pass to the time function. Defaults to None.
    '''
    # Set the configurations
    if log_conf is not None: 
        configure_logging(log_conf)
    if isinstance(prediction_conf, PredictionConfig):
        agents.prediction_config = prediction_conf
    
    # Set the get_time function
    if get_time_func is not None:
        if get_time_func_params is None:
            get_time_func_params = []
        agents.get_time = lambda: get_time_func(*get_time_func_params)
    else:
        agents.get_time = default_get_time_func
    
    agents.interface_jid = interface_jid

    try:
        spade.run(main(main_agent_jid, main_agent_password))
    except KeyboardInterrupt:
        logger.info('KeyboardInterrupt')
    except Exception as e:
        logger.error(f'Error: {e}')
    
    logger.info('All agents finished')

def system_start_tcp(tcp_interface_jid: str, 
                     tcp_interface_password: str, 
                     tcp_interface_host_ip: str,
                     tcp_interface_port: int,
                     main_agent_jid: str, 
                     main_agent_password: str, 
                     log_conf: Optional[dict|int] = logging.INFO,
                     prediction_conf: Optional[PredictionConfig] = None,
                     get_time_func: Optional[Callable[[], TimeSlot]] = None, 
                     get_time_func_params: Optional[Iterable[Any]] = None):
    '''Start the MAS system with the given configurations. This call blocks until the system is finished.
    This version of the function uses a TCP relay agent to communicate with other agents as a substitute for a XMPP interface.
    To communicate with the system send pickled spade messages to the TCP relay agent socket.

    Args:
        tcp_interface_jid (str): JID of the tcp agent
        tcp_interface_password (str): password of the tcp agent
        tcp_interface_host_ip (str): IP address of the host machine
        tcp_interface_port (int): Port number to bind the TCP relay agent to
        main_agent_jid (str): JID of the main agent
        main_agent_password (str): password of the main agent
        log_conf (Optional[dict | int], optional): Logging configuration used in logging module. Defaults to logging.INFO.
        prediction_conf (Optional[PredictionConfig], optional): Configuration of the prediction model. Defaults to None.
        get_time_func (Optional[Callable[[], TimeSlot]], optional): Function for getting the current environment time. If None, defaults to using datetime.
        get_time_func_params (Iterable[Any], optional): Prameters to pass to the time function. Defaults to None.
    '''
    # Set the configurations
    if log_conf is not None: 
        configure_logging(log_conf)
    if isinstance(prediction_conf, PredictionConfig):
        agents.prediction_config = prediction_conf
    
    # Set the get_time function
    if get_time_func is not None:
        if get_time_func_params is None:
            get_time_func_params = []
        agents.get_time = lambda: get_time_func(*get_time_func_params)
    else:
        agents.get_time = default_get_time_func
    
    agents.interface_jid = tcp_interface_jid


    # Run the system
    try:
        spade.run(tcp_main(main_agent_jid, 
                        main_agent_password, 
                        tcp_interface_jid, 
                        tcp_interface_password, 
                        tcp_interface_host_ip, 
                        tcp_interface_port))
    except KeyboardInterrupt:
        logger.info('KeyboardInterrupt')
    except Exception as e:
        logger.error(f'Error: {e}')
    
    logger.info('All agents finished')