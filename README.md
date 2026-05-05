# Impact of Image Degradation on Melanoma Detection by AI: A Patient Safety Perspective

This repository contains the source code for the study on AI vulnerability in skin cancer screening under real-world photographic conditions.

## Project Structure
*   **`dataset_loader.py`**: Custom PyTorch Dataset class for HAM10000.
*   **`model_training.py`**: Fine-tuning script for ResNet-50 and MobileNetV2.
*   **`robustness_stress_test.py`**: Logic for simulating blur, noise, and low luminance.
*   **`image_degradation_evaluator.py`**: Main script to generate the comparative results table.

## Installation
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Download the [HAM10000 dataset](https://doi.org/10.7910/DVN/DBW86T) and place it in the `data/` folder.

## Usage
1. **Train models**: `python model_training.py`.
2. **Run comparison**: `python comparacion_final.py`.

