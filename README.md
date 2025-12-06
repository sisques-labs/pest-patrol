# Pest Patrol 🐛🔍

A scalable and modular deep learning framework for detecting and classifying plant pests and diseases using computer vision. Built with PyTorch, this project provides a clean, production-ready solution for agricultural monitoring and plant health assessment.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Dataset Organization](#dataset-organization)
- [Configuration](#configuration)
- [Training](#training)
- [Prediction](#prediction)
- [Available Models](#available-models)
- [Advanced Features](#advanced-features)
- [Monitoring Training](#monitoring-training)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Pest Patrol is an AI-powered system designed to help farmers, gardeners, and agricultural researchers identify plant pests and diseases from images. The framework uses state-of-the-art deep learning models to classify various plant health conditions, enabling early detection and intervention.

### Key Capabilities

- **Multi-class Classification**: Classify multiple pest and disease categories simultaneously
- **Modular Architecture**: Clean, scalable codebase that's easy to extend
- **Flexible Configuration**: YAML-based configuration system for easy customization
- **Production Ready**: Includes checkpointing, logging, and monitoring tools
- **Multiple Architectures**: Support for various CNN architectures (ResNet, EfficientNet, MobileNet, etc.)

## ✨ Features

### Core Features

- **Modular Design**: Clean separation of concerns with dedicated modules for data, models, training, and utilities
- **Multiple Model Architectures**: Easy switching between ResNet, EfficientNet, MobileNet, and more
- **Flexible Configuration**: YAML-based configuration system with automatic path resolution
- **Data Augmentation**: Comprehensive augmentation strategies to improve model generalization
- **Training Utilities**: Built-in checkpointing, early stopping, and learning rate scheduling
- **Easy Inference**: Simple command-line interface for making predictions

### Advanced Features

- **Mixed Precision Training**: Faster training with reduced memory usage
- **Gradient Clipping**: Prevents gradient explosion during training
- **Multiple Loss Functions**: Support for Cross Entropy, Focal Loss, and Label Smoothing
- **Learning Rate Scheduling**: Cosine, Step, and Plateau schedulers
- **Early Stopping**: Automatic training termination based on validation metrics
- **TensorBoard Integration**: Real-time training visualization
- **SSL Handling**: Automatic handling of SSL certificate issues (especially on macOS)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- (Optional) CUDA-capable GPU for faster training

### Step-by-Step Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd pest-patrol
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv

   # On macOS/Linux:
   source venv/bin/activate

   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Verify Installation

Test that everything is set up correctly:

```bash
python -c "from src.models import create_model; print('✓ Installation successful!')"
```

## 📁 Project Structure

```
pest-patrol/
├── config.yaml              # Main configuration file
├── train.py                 # Training script
├── predict.py               # Prediction script
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── LICENSE                 # License file
│
├── data/                   # Data directory
│   ├── raw/                # Raw dataset (organized by class)
│   │   ├── class1/
│   │   │   ├── img1.jpg
│   │   │   └── img2.jpg
│   │   └── class2/
│   │       └── ...
│   └── processed/          # Processed data (auto-generated)
│
├── src/                    # Source code
│   ├── config/             # Configuration management
│   │   ├── __init__.py
│   │   └── config.py
│   │
│   ├── data/               # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── dataset.py      # Dataset classes
│   │   └── transforms.py   # Image transforms and augmentation
│   │
│   ├── models/             # Model definitions
│   │   ├── __init__.py
│   │   ├── base.py         # Base model class
│   │   └── factory.py      # Model factory for creating architectures
│   │
│   ├── training/           # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py      # Main trainer class
│   │   ├── losses.py       # Loss function implementations
│   │   └── metrics.py      # Evaluation metrics
│   │
│   └── utils/              # Helper functions
│       ├── __init__.py
│       ├── logging.py      # Logging utilities
│       ├── checkpoints.py  # Checkpoint management
│       └── optimizers.py   # Optimizer and scheduler creation
│
└── outputs/                # Output directory
    ├── checkpoints/        # Model checkpoints
    ├── logs/               # Training logs and TensorBoard files
    └── predictions/        # Prediction outputs
```

## 🏃 Quick Start

### 1. Prepare Your Dataset

Organize your images in the following structure:

```
data/raw/
├── Apple___Apple_scab/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Apple___healthy/
│   └── ...
└── Tomato___Bacterial_spot/
    └── ...
```

Each subdirectory represents a class (pest or disease type).

### 2. Configure Your Training

Edit `config.yaml` to set your preferences:

```yaml
model:
  name: "resnet50" # Choose your architecture
  pretrained: true # Use ImageNet pretrained weights

training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.001
```

### 3. Train the Model

```bash
python train.py
```

The training script will:

- Load and preprocess your dataset
- Split it into train/validation/test sets
- Train the model with your configuration
- Save checkpoints and logs automatically
- Display training progress and metrics

### 4. Make Predictions

```bash
# Single image
python predict.py path/to/image.jpg

# Multiple images
python predict.py path/to/images/ --top-k 3

# Save predictions to JSON
python predict.py image.jpg --output predictions.json
```

## 📊 Dataset Organization

### Directory Structure

Your dataset should be organized with one folder per class:

```
data/raw/
├── ClassName1/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── ClassName2/
│   └── ...
└── ClassNameN/
    └── ...
```

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

### Dataset Splitting

The framework automatically splits your dataset into:

- **Training set**: 80% (default, configurable)
- **Validation set**: 10% (default, configurable)
- **Test set**: 10% (default, configurable)

Splitting is stratified to ensure balanced distribution across classes.

## ⚙️ Configuration

The `config.yaml` file controls all aspects of training and inference. Here are the main sections:

### Paths Configuration

```yaml
paths:
  raw_data_dir: "data/raw"
  outputs_dir: "outputs"
  checkpoints_dir: "outputs/checkpoints"
  logs_dir: "outputs/logs"
```

### Model Configuration

```yaml
model:
  name: "resnet50" # Architecture name
  pretrained: true # Use pretrained weights
  dropout: 0.5 # Dropout rate
```

**Available models:**

- ResNet: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- EfficientNet: `efficientnet_b0` through `efficientnet_b4`
- MobileNet: `mobilenet_v3_small`, `mobilenet_v3_large`

### Training Configuration

```yaml
training:
  batch_size: 32 # Batch size
  num_epochs: 50 # Number of epochs
  learning_rate: 0.001 # Initial learning rate
  optimizer: "adam" # Optimizer (adam, sgd, adamw)
  scheduler: "cosine" # LR scheduler (cosine, step, plateau, none)
  mixed_precision: true # Use mixed precision training
  gradient_clip: 1.0 # Gradient clipping value
```

### Data Augmentation

```yaml
augmentation:
  train:
    horizontal_flip: true
    rotation_range: 15
    brightness_range: [0.8, 1.2]
    contrast_range: [0.8, 1.2]
    saturation_range: [0.8, 1.2]
    random_crop: true
```

### Loss Functions

```yaml
loss:
  name: "cross_entropy" # Options: cross_entropy, focal_loss, label_smoothing
  label_smoothing: 0.1 # For label_smoothing loss
  focal_alpha: 0.25 # For focal_loss
  focal_gamma: 2.0 # For focal_loss
```

## 🎓 Training

### Basic Training

Train with default configuration:

```bash
python train.py
```

### Custom Configuration File

Use a different config file:

```bash
CONFIG_PATH=my_config.yaml python train.py
```

### Training Features

#### Automatic Checkpointing

The framework automatically saves:

- **Best model**: Based on validation loss (saved as `best_model.pth`)
- **Periodic checkpoints**: Every N epochs (configurable)
- **Training state**: Includes optimizer state for resuming

#### Early Stopping

Configure early stopping in `config.yaml`:

```yaml
training:
  early_stopping:
    enabled: true
    patience: 10 # Stop after 10 epochs without improvement
    min_delta: 0.001 # Minimum change to qualify as improvement
    monitor: "val_loss" # Metric to monitor
```

#### Learning Rate Scheduling

Multiple schedulers available:

- **Cosine Annealing**: Gradually decreases LR following a cosine curve
- **Step**: Reduces LR by a factor at fixed intervals
- **Reduce on Plateau**: Reduces LR when a metric stops improving

#### Mixed Precision Training

Enabled by default for faster training and reduced memory usage. Disable if you encounter issues:

```yaml
training:
  mixed_precision: false
```

### Training Output

The training script generates:

1. **Console Logs**: Real-time training progress
2. **Log File**: Detailed logs saved to `outputs/logs/training.log`
3. **TensorBoard Logs**: Visualization data in `outputs/logs/tensorboard/`
4. **Checkpoints**: Model snapshots in `outputs/checkpoints/`
5. **Class Mapping**: JSON file mapping class names to indices

## 🔮 Prediction

### Basic Usage

```bash
python predict.py path/to/image.jpg
```

### Command-Line Options

```bash
python predict.py IMAGE_PATH [OPTIONS]

Arguments:
  IMAGE_PATH              Path to image file or directory

Options:
  --checkpoint PATH       Path to model checkpoint (default: best_model.pth)
  --config PATH           Path to config file (default: config.yaml)
  --top-k N               Show top N predictions (default: 5)
  --output PATH           Save predictions to JSON file
```

### Examples

**Single image prediction:**

```bash
python predict.py data/test/image.jpg
```

**Batch prediction:**

```bash
python predict.py data/test/ --top-k 3
```

**Use specific checkpoint:**

```bash
python predict.py image.jpg --checkpoint outputs/checkpoints/checkpoint_epoch_20.pth
```

**Save predictions:**

```bash
python predict.py data/test/ --output my_predictions.json
```

### Output Format

For single images, predictions are displayed as:

```
Predictions for image.jpg:
--------------------------------------------------
  Apple___Apple_scab: 95.32%
  Apple___Black_rot: 3.21%
  Apple___healthy: 1.47%
```

For batch predictions or when using `--output`, results are saved as JSON:

```json
{
  "path/to/image1.jpg": [
    ["Class1", 0.95],
    ["Class2", 0.03],
    ["Class3", 0.02]
  ],
  "path/to/image2.jpg": [...]
}
```

## 🤖 Available Models

### ResNet Family

- **ResNet-18**: Fast, lightweight model
- **ResNet-34**: Balanced performance
- **ResNet-50**: Good accuracy/speed tradeoff (recommended)
- **ResNet-101**: Higher accuracy
- **ResNet-152**: Maximum accuracy

### EfficientNet Family

- **EfficientNet-B0**: Fastest, smallest
- **EfficientNet-B1-B4**: Increasing accuracy and size
- Best balance of accuracy and efficiency

### MobileNet Family

- **MobileNet-V3 Small**: Optimized for mobile devices
- **MobileNet-V3 Large**: Better accuracy, still mobile-friendly

### Choosing a Model

- **For accuracy**: ResNet-101, EfficientNet-B4
- **For speed**: ResNet-18, MobileNet-V3 Small, EfficientNet-B0
- **For balance**: ResNet-50, EfficientNet-B2 (recommended starting point)

## 🔬 Advanced Features

### Custom Loss Functions

#### Focal Loss

Useful for imbalanced datasets:

```yaml
loss:
  name: "focal_loss"
  focal_alpha: 0.25
  focal_gamma: 2.0
```

#### Label Smoothing

Reduces overconfidence:

```yaml
loss:
  name: "label_smoothing"
  label_smoothing: 0.1
```

### Advanced Data Augmentation

The framework supports comprehensive augmentation:

- Random cropping and resizing
- Horizontal flipping
- Random rotation
- Brightness, contrast, and saturation adjustments
- Color jittering

Configure in `config.yaml`:

```yaml
augmentation:
  train:
    horizontal_flip: true
    rotation_range: 30 # ±30 degrees
    brightness_range: [0.7, 1.3]
    contrast_range: [0.7, 1.3]
    saturation_range: [0.7, 1.3]
    random_crop: true
```

### Multi-GPU Training

For systems with multiple GPUs, modify device configuration:

```yaml
device:
  use_gpu: true
  gpu_ids: [0, 1, 2, 3] # Use multiple GPUs
```

Note: Multi-GPU support requires manual implementation in the trainer (not yet included in current version).

### Resume Training

To resume from a checkpoint:

1. Load the checkpoint in your training script
2. Update `config.yaml` with appropriate starting epoch
3. The framework will continue from where it left off

## 📈 Monitoring Training

### TensorBoard

Visualize training progress in real-time:

```bash
tensorboard --logdir outputs/logs/tensorboard
```

Then open `http://localhost:6006` in your browser.

### Metrics Tracked

- Training loss and accuracy
- Validation loss and accuracy
- Learning rate schedule
- Batch-level metrics (configurable interval)

### Log Files

Detailed logs are saved to `outputs/logs/training.log` with:

- Epoch summaries
- Batch-level progress
- Best model information
- Error messages and warnings

## 🐛 Troubleshooting

### Common Issues

#### SSL Certificate Errors (macOS)

The framework automatically handles SSL certificate issues. If you encounter problems:

1. The code should handle it automatically
2. If issues persist, ensure you have Python's certificates installed:
   ```bash
   /Applications/Python\ 3.x/Install\ Certificates.command
   ```

#### Out of Memory Errors

- Reduce `batch_size` in `config.yaml`
- Use a smaller model (e.g., ResNet-18 instead of ResNet-50)
- Disable mixed precision: `mixed_precision: false`
- Reduce image size in dataset configuration

#### Slow Training

- Enable mixed precision training (default)
- Use GPU if available
- Reduce batch size if running out of memory
- Use data loading with multiple workers (already configured)

#### Poor Model Performance

- Increase training epochs
- Adjust learning rate
- Try different augmentation strategies
- Use a larger model architecture
- Check data quality and class balance

### Getting Help

1. Check the logs in `outputs/logs/training.log`
2. Review TensorBoard metrics for training patterns
3. Verify your dataset organization matches the expected structure
4. Ensure all dependencies are correctly installed

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

For development, install additional tools:

```bash
pip install black flake8 mypy pytest
```

## 📄 License

See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [PyTorch](https://pytorch.org/) and [torchvision](https://pytorch.org/vision/)
- Model architectures from [torchvision](https://github.com/pytorch/vision) and [timm](https://github.com/rwightman/pytorch-image-models)
- Inspired by the need for accessible agricultural AI tools

## 📚 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [torchvision Models](https://pytorch.org/vision/stable/models.html)
- [Transfer Learning Guide](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

---

**Made with ❤️ for the agricultural community**

For questions, issues, or feature requests, please open an issue on GitHub.
