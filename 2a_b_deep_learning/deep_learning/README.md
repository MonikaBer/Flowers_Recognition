# Deep learning
- This module includes source code for 2a. and 2b. exercise: 
2a) last convolutional layer learning
2b) next two last and last convolutional layers learning

- Directory named 'outputs' contains results of deep learning.

- Directory named 'plots' contains plots which show the results of deep learning.

- Directory named 'plots_drawer' contains source code for generating plots.

## Settings for deep learning
Source file named '2a_b_deep_learning.py' can be modified for execute the learning process:

If you want to learn only the last convolutional layer, you should uncomment in the main function line with setup_2a() 
function calling and you should comment line with setup_2b() function calling. Analogously, if you want to learn two 
convolutional layers (2b).

You should set proper path of flowers directory and path for saving the model too. These settings can be set by global 
variables named 'DATA_DIR' and 'SAVING_PATH'.

## Way to draw plots from results of deep learning
In result of deep learning the history of learning will be printed to standard output. That history can be used for
plot drawing (it should be saved in file and put as input to the plots drawer).
