# SmartHome-Inhabitant-Simulation
run simulation (from outside the project folder):  
<code>python3 -m \<folder name></code>  
Show help:  
<code>python3 -m \<folder name> -h</code>  

Simulation parameters can be adjusted via command line args or constants in Simulation.py

## Files  
<code>utils.py</code>: Helper functions  
<code>inhabitantModel.py</code>: Inhabitant base class  
<code>environment.py</code>: Environment with Timeslots, Home model and event system (derived from simpy.environment)  
<code>homeModel.py</code>: classes for modeling a smart home (rooms with devices)  
<code>deviceModels.py</code>: classes modeling smart devices
<code>event.py</code>: event system (publish, subscribe)  
<code>stateLogger.py</code>: Logs the environment state and inhabitants location and actions in given time intervals (logs used for machine learning)  
<code>inhabitants/*.py</code>: Inhabitants with concrete behaviors  
<code>Simulation.py</code>: Simulation experiment with some command line args
