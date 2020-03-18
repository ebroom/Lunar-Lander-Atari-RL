## How to Run
This project is using Python 3.4, it also uses the gym package as well as keras, tensorflow, numpy, and matplotlib.   
Simply clone the repo and run `python3 Main.py` This will train the agent on parameters epsilon-decay: .998, minimum-epsilon: 0.1, 
gamma: .999, alpha: .0001, maximum-memory: 65536, minimum-memory: 64, and batch-size: 32. If the agent succeeds at getting an average
over 200 in 100 episodes, the agent will then be tested on 100 more episodes. The program will output a csv for the training: `lunarlander_alpha_<alpha>_gamma_<gamma>_epislon_<min epsilon>.csv` and
`lunarlander_test_alpha_<alpha>_gamma_<gamma>_epislon_<min epsilon>.csv` for the testing. (I am aware I spelled epsilon wrong in the file names.) In these csvs are the rewards for every
episode separated by commas. They will be used in the Plotter.   
You can run the plots by `python3 Plotter.py`. It will produce plots for all the graphs, the rewards during training, the testing
graph with 100 episodes, and the variable alpha, gamma, and minimum epsilon plots. (final_testing.png, final_training.png, 
variable_alpha.png, variable_gamma.png, and variable_min_epsilon.png)   
Which, if you need to view the plots referenced in the paper, these are the images you will need.  
You can also uncomment the `env.render()` if you want to see the agent learn/test in real time.

## Running with Different Parameters
If you want to run the agent with different parameters, add another object to `parameters` in the `run` section of `Main.py`. Update 
the parameters given with the parameters you want (the order is listed in the same order as the parameters written in the first section).
If you want to update the neural network parameters (besides alpha) you will need to manually update in `NN.py`.
