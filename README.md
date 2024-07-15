# Learning Rate Visualizations
This repository contains a Python program that demonstrates the use of different learning rate schedulers in training a simple convolutional neural network (CNN) on the CIFAR-10 dataset using PyTorch.
RECOMMENDED to use a GPU with CUDA support and at least 6 GB VRAM

![learning_rate_CosineAnnealingWarmRestarts](https://github.com/user-attachments/assets/6c0a276f-5333-4f81-b921-0ebcca86ea81)


## Prerequisites
Before you can run this program, you'll need to install Python and PyTorch along with some other required libraries. It's recommended to use a virtual environment.

### Installation
1. Clone this repository
   ```bash
   git clone https://github.com/drewmeyerZE/learning_rate_visualizations.git
   cd learning_rate_visualizations
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   
## Running the Experiment
To run the experiment with different learning rate schedulers, execute the main.py file:
```bash
python visualizer.py
```
### Important Configurations
The batch size, learn rates, and number of epochs are edited in the config.py file
Graphs showing the learning rate schedules are saved to the project directory with names `learning_rate_{scheduler_name}.png` as defined on line 164. Change the paths if you prefer different storage locations.

## Dataset
This program uses the CIFAR-10 dataset, which is automatically downloaded when you run the program.
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.
See the the website [https://www.cs.toronto.edu/~kriz/cifar.html]

## Outputs
The program will output graphs for each learning rate scheduler showing the progression of learning rates over epochs. These will be saved in the project directory. Console logs will provide loss and accuracy metrics during training and evaluation.

## Contributing
Feel free to fork this repository and submit pull requests to enhance the functionalities of this experiment.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
