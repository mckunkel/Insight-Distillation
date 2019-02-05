# Insight-Distillation
Insight Project for Knowledge Distillation

Insight Project of Distilling an Neural Network, which is a method to transfer knowledge from a larger teacher model into a student model
For a indepth video lecture by Geoffrey Hinton see
https://www.youtube.com/watch?v=EK61htlw8hY
based upon the idea of [Geoffrey Hinton's paper](https://arxiv.org/abs/1503.02531).
An indepth video, given by Geoffrey Hinton can be seen [here](https://www.youtube.com/watch?v=EK61htlw8hY), with slides to the talk located [here](http://www.ttic.edu/dl/dark14.pdf).

## Usage
* It's recommended to use an environment
** Using Anaconda
*** [Install Anaconda](https://www.anaconda.com/distribution/)
*** conda create -n DistillingNeuralNetwork python=3.6
*** conda install keras matplotlib numpy pandas Pillow torchvision tqdm

### Anaconda Environment
* see [Setting a conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
* conda environment included as environment.yml
*  conda env create -f environment.yml
* conda activate Distilling


## Usage
### Data (folder data_stuff)
#### Get the Data
* python3 run_scripts.py
#### This will take time depedning on your internet connection
### Split the data
* python3 split_and_save_images.py
#### This will take some time depending on your configuration

### Get a new set of weights with the CalTech256 image data set
* python3 train_xception.py
* python3 get_logits.py
** This should also save several metric plots
*** top5_accuracy_vs_epoch.png
*** accuracy_vs_epoch.png
*** logloss_vs_epoch.png


