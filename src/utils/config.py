import os 
import torch

class Config:

    # Models
    # "prajjwal1/bert-tiny": 2 layers / 4.4M parameters / out 128 dimensions
    # "prajjwal1/bert-mini": 4 layers / 11.3M parameters / out 256 dimensions
    # "prajjwal1/bert-small": 4 layers / 29.1M parameters / out 512 dimensions
    # "bert-base-uncased": 12 layers / 110M parameters / out 768 dimensions

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "prajjwal1/bert-mini"  
    NUM_FROZEN_LAYERS = 2
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 5
    MARGIN = 0.2
    BATCH_SIZE = 32
    MAX_LENGTH = 256
    TEST_SIZE = 0.2

    # Memory management
    BATCH_SIZE = 16  # Reduced from 32
    GRADIENT_ACCUMULATION_STEPS = 4
    USE_CPU_IF_OOM = False

    # Model saving parameters
    MODEL_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    SAVE_INTERMEDIATE_MODELS = True
    SAVE_INTERVAL = 1  # Save every N epochs
    
    # Dataset
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    INPUT_DIRECTORY = os.path.join(BASE_DIR, "data", "raw")
    CSV_FILE = os.path.join(BASE_DIR, "data", "processed", "literary_texts.csv")
    TRIPLETS_FILE = os.path.join(BASE_DIR, "data", "processed", "triplets.json")
    TOKENIZED_TRIPLETS_FILE = os.path.join(BASE_DIR, "data", "processed", "tokenized_triplets.pt")
    MIN_BLOCK_SIZE = 300
    MAX_BLOCK_SIZE = 1000

    # Logging
    LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    LOG_LEVEL = "INFO"