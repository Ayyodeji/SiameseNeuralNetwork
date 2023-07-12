# Siamese Neural Network for Gene Image Similarity Comparison

This repository contains an implementation of a Siamese neural network used for gene image similarity comparison. The code includes functions for constructing the network, training the model, and performing gene image similarity predictions.

## Repository Structure

- `main.py`: The main script for training the Siamese neural network model and evaluating its performance.
- `siamese_network.py`: Contains functions for constructing the Siamese network architecture and defining the triplet loss function.
- `distance_layer.py`: Defines the DistanceLayer class used to compute the Euclidean distances between encoded feature vectors.
- `utils.py`: Includes utility functions for data preprocessing and loading the dataset.
- `singletest.py`: Provides a function for predicting gene image similarity on a single input gene image.

## Installation and Setup

To use this code, follow these steps:

1. Clone the repository:
git clone https://github.com/Ayyodeji/SiameseNeuralNetwork.git
cd SiameseNeuralNetwork


2. Set up the Python environment:
- Make sure you have Python 3.x installed.
- Create a virtual environment (optional but recommended):
  ```
  python -m venv env
  source env/bin/activate  # For Linux/Mac
  env\Scripts\activate  # For Windows
  ```
- Install the required dependencies:
  ```
  pip install -e .
  ```

## Usage

### Training the Siamese Model

To run the Siamese neural network model, run the following command:
python main.py

