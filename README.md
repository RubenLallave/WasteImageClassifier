# Waste Image Classifier

This project uses a deep learning model to classify waste images, aiming to improve waste management and recycling. It employs TensorFlow and Keras for model development and training, along with other Python libraries for data processing and visualization.

**This project was developed for academic purposes. Feel free to explore, use, and adapt the code and findings for your own learning or non-commercial applications.**

## Overview

The goal is accurate classification of waste images (plastic, paper, glass, etc.). The project includes:

- Data Handling: Loading, preprocessing, and augmentation.
- Model Building: CNN implementation (potentially ResNet-based).
- Training: Notebooks and scripts for model training.
- Evaluation: Performance assessment using standard metrics.
- Inference: Classifying new waste images.
- Notebooks: Detailed project stages in Jupyter notebooks.
- Models: Trained model files (managed with Git LFS).
- Slides: Project summary presentation.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow (PIL)
- Scikit-image
- Requests
- python-dotenv
- Pexels and Pixabay APIs
- Git
- Git LFS

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/RubenLallave/WasteImageClassifier.git](https://github.com/RubenLallave/WasteImageClassifier.git)
    cd WasteImageClassifier
    ```

2.  **Install Git LFS:**
    ```bash
    git lfs install
    git lfs pull
    ```

3.  **Create and activate virtual environment (recommended):**
    ```bash
    conda create --name tf_python39 python=3.9 && conda activate tf_python39
    # or
    python -m venv venv && source venv/bin/activate  # Linux/macOS
    # or
    python -m venv venv && venv\Scripts\activate  # Windows
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

- Explore notebooks in the `notebooks/` directory for data and model development.

## Model Information

- **Model Architecture:** The image classification model is based on the **ResNet50** architecture. We leveraged the pre-trained weights of ResNet50 as a feature extractor and added custom fully connected layers on top for the specific task of waste classification. These custom layers include a Flatten layer, one or more Dense layers (potentially with Batch Normalization and Dropout for regularization), and a final Dense output layer with a **softmax activation function** to predict the probability distribution over the waste classes.

- **Number of Classes:** The model was trained to classify waste images into 10, 7 and 3 distinct categories.

- **Training Details and Performance:** The model was trained on several datasets of waste images (including API's and Google images), split into training and validation sets. We employed techniques such as **data augmentation** (e.g., random rotations, flips, zooms) to improve the model's generalization ability. The model was trained using the **Adam optimizer** with a chosen learning rate, and the **categorical cross-entropy loss function** was used to measure the error during training. We monitored the model's performance on the validation set using metrics like **accuracy**, **precision**, **recall**, and **F1-score**, and potentially used callbacks like **Early Stopping** and **ReduceLROnPlateau** to optimize the training process and prevent overfitting. Further details on the training process and performance can be found in the Jupyter notebooks within the `notebooks/` directory.

## Presentation

See `slides/` directory.

## Contributing

Feel free to fork and submit pull requests.
