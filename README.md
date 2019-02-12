# Insight-Distillation
Insight Project for Knowledge Distillation

Insight Project of Distilling an Neural Network, which is a method to transfer knowledge from a larger teacher model into a student model
For a indepth video lecture by Geoffrey Hinton see
https://www.youtube.com/watch?v=EK61htlw8hY
based upon the idea of [Geoffrey Hinton's paper](https://arxiv.org/abs/1503.02531).
An indepth video, given by Geoffrey Hinton can be seen [here](https://www.youtube.com/watch?v=EK61htlw8hY), with slides to the talk located [here](http://www.ttic.edu/dl/dark14.pdf).

## Usage
* It's recommended to use an environment
### Using Anaconda
* [Install Anaconda](https://www.anaconda.com/distribution/)
* conda create -n DistillingNeuralNetwork python=3.6
* conda install keras matplotlib numpy pandas Pillow torchvision tqdm

### Data 
* folder data_stuff
#### Get, decode, split data into valdation and training set
```
python3 run_scripts.py
```
##### This will take time depending on your internet connection and your configuration
### Modeling
#### Get a new set of weights with the CalTech256 image data set
```
 python3 train_xception.py
``` 
#### Get the logits from the Xception model and store them for both the training and validation dataset
```
python3 get_logits.py
```
#### Create a model and save it in the folders models
##### examples are in the folder models
* microXception.py
* squeezenet.py
* mobilenet.py
#### Distill the student model
```
distill_student.py -t <temperature> -l <lambda> -s <save name>
```
##### Be sure to change the import
* line 17: from models.squeezenet import SqueezeNet, preprocess_input
* line 66: model = SqueezeNet(weight_decay=1e-4, image_size=299)
#### For comparison run your model without distillation
#### This should also save several metric plots
* top5_accuracy_vs_epoch.png
* accuracy_vs_epoch.png
* logloss_vs_epoch.png
* model\_distilled\_\<save name\>\_model\_T\_\<temperature\>\_lambda\_\<lambda\>.h5
### Running on an Edge device
#### Instructions for a Raspberry Pi 3 B+
* set up the [Raspberry Pi](https://projects.raspberrypi.org/en/projects/raspberry-pi-setting-up)
* Set up [Camera](https://www.raspberrypi.org/documentation/configuration/camera.md) for Raspberry Pi
* Set up [SSH](https://www.raspberrypi.org/documentation/remote-access/ssh/) on the Raspberry Pi
* Set up virtual environment on the pi and use raspberry_pi_environment.txt file in raspberry_pi_stuff to have the correct environment for the Raspberry Pi
```
sudo pip install virtualenv virtualenvwrapper 
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
source /usr/local/bin/virtualenvwrapper.sh
source ~/.profile
mkvirtualenv <env_name> -p python3
workon <env_name>
(<env_name>)$ pip install -r path/to/raspberry_pi_environment.txt
```
* Use the script pi_running.py in raspberry_pi_stuff as template to run the model



