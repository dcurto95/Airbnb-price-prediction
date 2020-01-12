********** HOW TO RUN THE CODE **********

For running this project you will need to install "Pytorch 1.3".
This project is composed by 5 main files, each one with its purpose:

 - csv_maker.py: This file is used to create the cleaned, preprocessed and splitted datasets that
 will be later used by the other main files. This was done to compare the different models with
 the same instances.
 
 - data_exploration.py: When executing this file, it will show all the graphs used to analyze the data.
 
 - feature_test.py: This python file was used to compare the metrics achieved with the different 
 preprocessing techniques. 
 
 And now the two most important files. Both of them have stablished the optimal configuration of 
 hyperparameters:
 
 - main_linear_regression.py: It executes the Linear regression and SVR models for the previously
 created data splits (by csv_maker.py). It will show the MSE, R2 and execution time for each
 feature configuration.
 
 - main_mlp.py: This other main will run the MLP model with the optimal configuration. It is 
 fixed to 1000 epochs maximum but you can see the training progress if you run in your operative 
 system console: "tensorboard --logdir=logs" from the project folder.
 
 All the needed .csv files will be provided so you don't have to create them again. They will all
 be located in the "data" directory.