# StancePrediction

This repository provides a reference implementation of debate stance prediction model as described in the paper:<br>
> Structured Representation Learning for Online Debate Stance Prediction.<br>
> Chang Li, Aldo Porco, Dan Goldwasser.<br>
> The 27th International Conference on Computational Linguistics (COLING 2018).<br>
> <Insert paper link>

### Basic Usage

#### Data and Saved Models

Download the dataset and saved models from https://figshare.com/projects/Debate_Stance_Prediction/35072. Uncompress the zip files and put 'data' and 'result' folders in the root directory.

#### Example
Run:

``python code\runner_4forum.py``

This program will load data from the 'data/' folder, construct (and train) embedding neural nets for post, author and stances on different topics. We used 5-fold cross validation to get the average accuracy. Trained models will be stored under the specified folder inside 'result/'.


