# Literary Style Analysis

This project implements a deep learning model for analyzing and comparing literary styles using a triplet network architecture based on BERT. It can be used to identify stylistic similarities between texts and potentially for tasks such as author attribution or style transfer.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Testing](#testing)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Features

- Triplet network architecture using BERT for text embedding
- Custom dataset creation from raw text files
- Tokenization and preprocessing of literary texts
- Training with triplet loss for style comparison
- Evaluation of model performance
- Similarity analysis between text samples
- Autonomous testing script for random triplet analysis and custom text comparison

## Project Structure

```
literary-style-analysis/
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── data_processing.py
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── model_utils.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── testing.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── logger.py
│
├── scripts/
│   └── run_training.py
│
├── tests/
│   └── (test files here)
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── (saved models here)
│
├── notebooks/
│   └── (Jupyter notebooks for analysis here)
│
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

## Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended for faster training)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/literary-style-analysis.git
   cd literary-style-analysis
   ```

2. Set up a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the package with CUDA support:
   ```
   pip install -e .[cuda] --find-links https://download.pytorch.org/whl/cu118
   ```

   This will install the CUDA-enabled version of PyTorch along with other dependencies.

   If you don't have CUDA, or wish to use the CPU-only version, you can install the dependencies without CUDA support:
   ```
   pip install -e .
   ```

4. Prepare your dataset:
   - Place your text files in the `data/raw` directory
   - Update the `INPUT_DIRECTORY` in `src/utils/config.py` if necessary
   - Run the data preparation script:
     ```
     python -m src.data.data_processing
     ```

Now you're ready to start training your model!

## Training

The training process can be initiated by running:

```
python scripts/run_training.py
```

This script handles data loading, model initialization, and the training loop. You can modify training parameters in `src/utils/config.py`.

## Testing

The project includes an autonomous testing script for evaluating the model. To use it:

1. For random triplet analysis:
   ```
   python -m src.evaluation.testing --random
   ```

2. For custom text comparison:
   ```
   python -m src.evaluation.testing --text1 "Your first text here" --text2 "Your second text here"
   ```

## Configuration

Adjust the settings in `src/utils/config.py` to customize:
- Model architecture (BERT variant)
- Training parameters (learning rate, batch size, etc.)
- Data processing settings
- Logging and model saving options

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.