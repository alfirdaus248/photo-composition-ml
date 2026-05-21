# Digital Photo Composition Assessment System

A Streamlit web application for assessing digital photo composition using MobileNetV2 transfer learning. The system predicts whether a photo has good composition or needs improvement, then provides visual feature-based recommendations.

## Features

- Upload and analyze a photo
- Predict photo composition quality
- Display prediction confidence
- Extract visual features from the image
- Provide rule-based improvement recommendations
- Download the analysis result as a text file

## Quick Start

The trained MobileNetV2 model is already included, so you can run the application without training.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/photo-composition-assessment.git
cd photo-composition-assessment
```

### 2. Create a Virtual Environment

This project was tested with Python 3.13.9.

For Windows:

```bash
py -3.13 -m venv venv
```

For macOS or Linux:

```bash
python3.13 -m venv venv
```

### 3. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

For macOS or Linux:

```bash
source venv/bin/activate
```

Check the Python version:

```bash
python --version
```

It should show Python 3.13.x. Python 3.11 or 3.12 should also work, but Python 3.14 or newer may cause TensorFlow installation issues.

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app_mobilenet.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Requirements

Recommended Python version:

```text
Python 3.11, 3.12, or 3.13
```

This project was tested with Python 3.13.9.

Main dependencies:

```txt
streamlit
tensorflow
numpy
opencv-python
pillow
scikit-learn
pandas
matplotlib
joblib
tqdm
```

## Model Files

The trained model files are included in this repository:

```text
model/
├── mobilenet_composition_model.h5
└── mobilenet_label_classes.npy
```

These files are required to run the application.

## Dataset

This project uses the CADB dataset:

**CADB - Image Composition Assessment Dataset**  
https://github.com/bcmi/Image-Composition-Assessment-Dataset-CADB

The dataset contains 9,497 images with composition scores rated by human annotators. The original scores are converted into two labels:

| Mean Score | Label |
|---|---|
| `< 3.0` | `perlu_perbaikan` |
| `>= 3.0` | `baik` |

The dataset is not included in this repository because of its large file size. It is only needed if you want to retrain the model.

## Project Structure

```text
photo-composition-assessment/
├── app_mobilenet.py
├── train_mobilenet.py
├── train_model_binary.py
├── extract_features.py
├── extract_features_binary.py
├── prepare_labels.py
├── prepare_labels_binary.py
├── check_dataset.py
├── requirements.txt
├── README.md
├── model/
│   ├── mobilenet_composition_model.h5
│   └── mobilenet_label_classes.npy
└── assets/
    ├── app_main.png
    └── app_recommendation.png
```

## How It Works

1. The user uploads a photo.
2. The image is resized and preprocessed.
3. MobileNetV2 predicts the composition class.
4. OpenCV extracts visual features from the image.
5. The system displays the result, confidence score, visual features, and recommendations.
6. The user can download the analysis result.

## Visual Features

| Feature | Description |
|---|---|
| Brightness | Average image brightness |
| Contrast | Difference between dark and bright areas |
| Sharpness | Blur estimation using Laplacian variance |
| Edge Density | Amount of detected edge details |
| Colorfulness | Color variation in the image |
| Rule of Thirds Score | Estimated visual elements near rule-of-thirds lines |
| Symmetry Score | Estimated left-right visual balance |
| Aspect Ratio | Width-to-height ratio |

## Results

| Model | Accuracy |
|---|---:|
| Random Forest with manual visual features | 67.79% |
| MobileNetV2 Transfer Learning | 74.73% |

Classification report for MobileNetV2:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| `baik` | 0.61 | 0.69 | 0.65 |
| `perlu_perbaikan` | 0.83 | 0.78 | 0.80 |

## Optional: Retrain the Model

Download the CADB dataset and place it in this structure:

```text
dataset/
└── CADB_Dataset/
    ├── composition_scores.json
    ├── composition_attributes.json
    ├── composition_elements.json
    ├── scene_categories.json
    └── images/
```

Check the dataset:

```bash
python check_dataset.py
```

Prepare labels:

```bash
python prepare_labels_binary.py
```

Train MobileNetV2:

```bash
python train_mobilenet.py
```

The trained model will be saved in:

```text
model/
├── mobilenet_composition_model.h5
└── mobilenet_label_classes.npy
```

## Limitations

- Photo composition is subjective.
- The system only uses two classes: `baik` and `perlu_perbaikan`.
- Recommendations are generated using rule-based visual features.
- The system does not yet detect photography genre or specific composition type.

## Future Development

- Photography genre classification
- Composition type detection
- Multi-class quality prediction
- Grad-CAM or saliency map visualization
- Improved recommendation generation
- Web deployment

## Author

M. Hisyam Al Firdaus

## License

This project is created for academic and educational purposes.