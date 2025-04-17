
# Sarcasm Detection with TensorFlow and Keras

A natural language processing (NLP) project that classifies text statements as sarcastic or non-sarcastic using deep learning.

## Project Overview

This project implements a binary text classification model that determines whether headlines or statements are sarcastic. The model uses word embeddings and neural networks to understand the semantic meaning behind text and identify sarcastic content.

## Dataset

The project uses the Sarcasm Detection dataset that contains news headlines labeled as sarcastic (1) or not sarcastic (0). Each entry in the dataset consists of:
- A headline
- A binary label indicating whether the headline is sarcastic

## Implementation Details

### Text Preprocessing
- Tokenization of text data using TensorFlow's Keras preprocessing
- Handling out-of-vocabulary (OOV) words
- Sequence padding to ensure uniform input length

### Model Architecture
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 100, 16)           160000    
_________________________________________________________________
global_average_pooling1d (Gl (None, 16)                0         
_________________________________________________________________
dense (Dense)                (None, 24)                408       
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 25        
=================================================================
Total params: 160,433
Trainable params: 160,433
Non-trainable params: 0
_________________________________________________________________
```

The model consists of:
- An embedding layer that converts tokenized words into dense vectors
- A global average pooling layer to reduce dimensionality
- A dense hidden layer with ReLU activation
- A final output layer with sigmoid activation for binary classification

### Training
- Binary cross-entropy loss function
- Adam optimizer
- 30 epochs with validation

## Results

The model achieves good accuracy on both training and validation datasets, with visualizations showing the learning trajectory over epochs.

Sample predictions:
- "game of thrones season finale showing this sunday night" → Not sarcastic
- "You are bald enough to have a haircut" → Sarcastic

## Technologies Used

- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Python 3.x

## How to Use

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib
   ```
3. Run the notebook or script:
   ```bash
   python sarcasm_detection.py
   ```
   or open and run the notebook in Jupyter/Colab

## Future Improvements

- Experiment with different model architectures (LSTM, CNN)
- Try different word embedding approaches (pre-trained GloVe, Word2Vec)
- Implement data augmentation techniques
- Add support for longer text analysis

## Acknowledgements

The sarcasm dataset is sourced from [Learning Datasets](https://storage.googleapis.com/learning-datasets/sarcasm.json).
