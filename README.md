# Literary Style Analysis

This project implements a deep learning model for analyzing and comparing literary styles using a triplet network architecture based on BERT. It can be used to identify stylistic similarities between texts and potentially for tasks such as author attribution or style transfer.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Training Pipeline](#training-pipeline)
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
- Flexible pipeline for data preparation and model training
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

## Usage

The main script for running the literary style analysis pipeline is `scripts/run_training.py`. This script now supports running different stages of the pipeline separately or together.

### Training Pipeline

The training pipeline consists of the following steps:

1. Create dataset CSV
2. Generate triplets
3. Tokenize triplets
4. Train the model
5. (Optional) Display random triplet analysis

You can run these steps individually or combine them as needed. Here are some example usages:

1. To create the CSV file only:
   ```
   python scripts/run_training.py --create_csv
   ```

2. To create triplets:
   ```
   python scripts/run_training.py --create_triplets 
   ```

3. To tokenize the triplet:
   ```
   python scripts/run_training.py  --tokenize_triplets
   ```

4. To train the model and display random triplet analysis:
   ```
   python scripts/run_training.py --train --display_random
   ```

5. To run the entire pipeline:
   ```
   python scripts/run_training.py --create_csv --create_triplets --tokenize_triplets --train --display_random
   ```

### Testing

You can use the testing script to evaluate the model on random triplets or compare custom texts:

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