# Flower Classifier to TensorFlow Lite

A flower image classifier built using **TFLite Model Maker** — trains on Google's Flower Photos dataset and exports a lightweight `.tflite` model ready for mobile and edge deployment. Designed to run in Google Colab.

---

## Flowers Classified

| Flower |
|---|
| 🌻 Sunflowers |
| 🌷 Tulips |
| 🌹 Roses |
| 🌼 Daisies |
| 🌸 Dandelions |

---

## Requirements

- Python 3.x
- TensorFlow
- TFLite Model Maker (`tflite-model-maker`)
- Google Colab

Install dependencies:

```bash
pip install tflite-model-maker
```

> ⚠️ **Note:** `tflite-model-maker` may fail on newer Python/TF versions due to compatibility issues. It is best run on **Google Colab** with TensorFlow 2.x.

---

## Pipeline

### Step 1 — Download Dataset
The Flower Photos dataset is downloaded automatically from Google's servers:

```python
data_path = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True
)
```

**Dataset:** ~3,600 images across 5 flower categories.

### Step 2 — Load & Split Data
Data is loaded from the folder structure and split:
- **90% Training**
- **10% Testing**

### Step 3 — Train Model
Uses TFLite Model Maker's `image_classifier.create()` which fine-tunes a MobileNet-based model under the hood:

```python
model = image_classifier.create(train_data, epochs=5)
```

### Step 4 — Evaluate
Prints loss and accuracy on the test set.

### Step 5 — Export to TFLite
Exports the trained model as a `.tflite` file and downloads it via Colab:

```python
model.export(export_dir='exported_model')
files.download('exported_model/model.tflite')
```

---

## Output

- `exported_model/model.tflite` — Optimized model for mobile/edge inference

---

## Usage

1. Open the notebook in **Google Colab**.
2. Run all cells in order.
3. The `.tflite` model file will be automatically downloaded after training.

---

## Dataset

**Source:** [Google Flower Photos](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)

**Classes:** Daisy, Dandelion, Roses, Sunflowers, Tulips

**Size:** ~3,600 images
