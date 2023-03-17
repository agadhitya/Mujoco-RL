# manipulation_planning

This package is used to read and execute planned manipulation sequences. Python scripts receive planning outputs and call rosservices for either simulated or real executions of manipulation primitives. Following nodes are used to advertise services.

### Nodes:

#### manip_acts

This node advertises services for each manipulation primitive in the simulation. Python scripts use these services when executing a manipulation sequence in simulation.

#### arm_adjust

This node advertises a service for arm adjustment. It is frequently used by the plan_ui, when setting the initial states for a simulated experiment.

#### franka_acts

This node advertises services for Franka arm motion on hardware. Services are called within the python scripts that read and execute manipulation sequences.
