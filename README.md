# Inhibitory-Synaptic-Plasticity
This repository contains code to model inhibitory plasticity in neural networks, along with simulations demonstrating its effects.

## About
Inhibitory neurons play a critical role in regulating neural network activity. This repository focuses on modeling spike timing dependent plasticity for inhibitory synapses.

The plasticity rule implements homeostatic regulation of postsynaptic firing rates. It encourages inhibition when rates are too high, and disinhibition when rates are too low.

## Getting Started
The main simulation code is in Python, using the PymoNNtorch framework. To run the simulations:

- Install dependencies including PymoNNtorch
- Clone the repository
- Run the Python notebooks for the desired simulation
- Each directory contains code for the different scenarios covered in the report.

## Simulations
- **Balanced Network:** Inhibitory plasticity helps balance excitation and inhibition in a randomly connected network
- **FeedForward Inhibition:** reduces overall input drive to a target postsynaptic neuron
- **Signal Learning:**
  - Unsupervised Learning with K-Winners-Take-All - clusters inputs into selectively responsive neural groups  
  - Reinforcement Learning - learns to respond to designated input patterns

  
See the report PDF for details on the simulation setups, parameters, and results.
## Contributing
Contributions are welcome! Please open issues for any bugs found or ideas for extensions. Pull requests to fix issues or add new simulation scenarios are appreciated.
