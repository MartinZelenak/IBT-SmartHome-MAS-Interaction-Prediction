# SmartHome-Inhabitant-Simulation
run simulation:  
<code>python3 Scenario.py</code>

inhabitantModel.py: Inhabitant base class  
environment.py: Environment with Timeslots (derived from simpy.environment)  
homeModel.py: classes for modeling a smart home (rooms with devices)  
Scenario.py: Simulation experiment. Inhabitant with concrete behavior (derived from inhabitantModel.Inhabitant)  
