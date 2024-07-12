# Learning Rate Visualizations
This repository contains a Python program that demonstrates the use of different learning rate schedulers in training a simple convolutional neural network (CNN) on the CIFAR-10 dataset using PyTorch.
RECOMMENDED to use a GPU with CUDA support and at least 6 GB VRAM

## Prerequisites
Before you can run this program, you'll need to install Python and PyTorch along with some other required libraries. It's recommended to use a virtual environment.

### Installation
1. Clone this repository
   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   
## Running the Experiment
To run the experiment with different learning rate schedulers, execute the main.py file:
```bash
python model.py
```
### Important Configurations
Here are some important lines in the code you might want to adjust based on your experimental setup:
- Batch Size: The batch size for training and evaluation is set on line 36. Modify this value to fit the memory capabilities of your GPU.
- Learning Rate Schedulers: Starting at line 94, the parameters for each learning rate scheduler can be adjusted. Each scheduler can be configured to see how it affects the training dynamics.
- Number of Epochs: The number of training epochs is set on line 129. Increase or decrease based on how long you want the training to run.
- Output Path for Graphs: Graphs showing the learning rate schedules are saved to the project directory with names `learning_rate_{scheduler_name}.png` as defined on line 148. Change the paths if you prefer different storage locations.

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
