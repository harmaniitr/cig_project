# Distorted Visual Sequence Pattern Recognition using Deep Learning

This project focuses on recognizing and reconstructing text sequences from distorted grayscale images using deep learning. The images contain character sequences affected by noise, blur, overlapping symbols, irregular spacing, shape deformation, and visual artifacts.

The goal of this project is to build a model that can read a distorted image and predict the correct ordered sequence of characters.

## Project Overview

In many real-world computer vision tasks, text is not always clear or properly aligned. Characters may be noisy, partially hidden, overlapped, or distorted. Traditional OCR methods may fail in such cases.

To solve this problem, I developed a CRNN-based deep learning model that combines:

* Convolutional Neural Networks (CNNs) for visual feature extraction
* Bidirectional LSTM layers for sequence learning
* Connectionist Temporal Classification (CTC) loss for sequence alignment
* CTC decoding for final text prediction

## Dataset

The dataset consists of grayscale distorted text sequence images.

The training set contains:

* Image files
* Corresponding text labels

The test set contains:

* Image files only

The model predicts the text sequence present in each test image and generates a final submission CSV file.

**Note:** The dataset is not uploaded in this repository due to file size limitations. Please place the training and testing image folders in the required paths before running the notebook.

## Approach

The project follows this pipeline:

* Load image paths and text labels from the training CSV file
* Convert text labels into character-level integer sequences using a tokenizer
* Pad labels to make them compatible for batch training
* Preprocess images by converting them to grayscale, normalizing pixel values, resizing, and transposing them
* Apply mild data augmentation such as brightness change, contrast change, and random noise
* Train a CRNN model using CNN, BiLSTM, and CTC loss
* Decode model predictions using CTC greedy decoding
* Evaluate the model using Character Error Rate and exact word match accuracy
* Generate the final prediction CSV file

## Model Architecture

The model follows a CRNN architecture.

CNN layers are used to extract visual features from the distorted images. After convolution and pooling, the feature map is reshaped into sequential timesteps along the horizontal direction of the image. Each timestep represents a vertical slice of visual features.

These sequential features are then passed into Bidirectional LSTM layers, which learn character dependencies from both left-to-right and right-to-left directions.

Finally, a dense softmax layer predicts character probabilities at each timestep. CTC loss is used because the exact alignment between image regions and output characters is unknown.

## Why CTC Loss?

In this task, the model only knows the complete label of each image, not the exact position of every character inside the image.

CTC loss is useful because it allows the model to learn the alignment between image features and output characters automatically. This avoids the need for manual character segmentation, which is difficult for distorted and overlapping characters.

## Reproducibility

To make the notebook reproducible, fixed random seeds were used for Python, NumPy, and TensorFlow. The train-validation split was also fixed using `random_state=42`.

ModelCheckpoint was used to save the best model based on validation loss.

## Evaluation Metrics

The model was evaluated using two metrics:

### Character Error Rate

Character Error Rate measures the character-level difference between the predicted sequence and the true sequence using edit distance. Lower CER indicates better performance.

### Exact Word Match Accuracy

Exact word match accuracy measures the percentage of images where the complete predicted sequence exactly matches the true label.

## Results

The final model achieved the following validation performance on 2000 validation images:

| Metric                     | Value  |
| -------------------------- | ------ |
| Character Error Rate (CER) | 0.68%  |
| Exact Word Match Accuracy  | 96.00% |

The low CER shows that the model makes very few character-level mistakes. The high exact match accuracy indicates that most complete distorted character sequences were predicted correctly.

## Experiments Tried

Several experiments were performed during development:

| Experiment                    | Observation                                                        |
| ----------------------------- | ------------------------------------------------------------------ |
| Direct image loading          | Not memory efficient for larger datasets                           |
| `tf.data` pipeline            | Improved batch-wise image loading and training efficiency          |
| Standard Adam optimizer       | Validation loss showed unstable fluctuations                       |
| Adam with AMSGrad             | Helped stabilize optimization during CTC training                  |
| Data augmentation             | Improved robustness against noisy and distorted images             |
| CTC as separate loss function | Gave lower validation performance in my experiment                 |
| Custom CTC layer              | Gave better validation performance and was kept in the final model |

## Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

## How to Run

1. Clone the repository:

```bash
git clone <your-repository-link>
cd <repository-name>
```

2. Install required libraries:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

3. Open the notebook:

```bash
jupyter notebook
```

4. Run the notebook cells from top to bottom.

5. The final prediction file will be generated in CSV format.

## Expected Folder Structure

```text
project-folder/
│
├── final.ipynb
├── train-labels.csv
├── train_images/
│   ├── train-0.png
│   ├── train-1.png
│   └── ...
├── test_images/
│   ├── test-0.png
│   ├── test-1.png
│   └── ...
└── README.md
```

## Output Format

The final submission file contains two columns:

```csv
image,prediction
test-0.png,AXU323
test-1.png,BT91KD
```

## Project Status

The project successfully builds an end-to-end distorted character recognition pipeline using a CRNN model with CTC loss. The final model achieved strong validation performance and generated predictions for unseen test images.
