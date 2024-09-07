import warnings
import time
import torch
import os
import sys
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import logger
from src.model.model import TripletLiteraryStyleNetwork
from src.data.dataset import LiteraryStyleDataset
from src.training.train import train_triplet, test_triplet, create_data_loader
from src.model.model_utils import load_model, get_model_path
from src.data.data_processing import create_dataset_csv, generate_and_save_triplets, tokenize_and_save_triplets
from src.evaluation.testing import display_random_triplet_similarities

# Set environment variables to suppress warnings
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub.file_download')
warnings.filterwarnings("ignore", category=FutureWarning, module='transformers.tokenization_utils_base')
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

def prepare_data():
    logger.info("Checking if data preparation is needed")
    if not os.path.exists(Config.TOKENIZED_TRIPLETS_FILE):
        logger.info("Tokenized triplets not found. Starting data preparation.")
        if not os.path.exists(Config.CSV_FILE):
            logger.info("CSV file not found. Creating dataset CSV.")
            create_dataset_csv(Config.INPUT_DIRECTORY, Config.CSV_FILE)
        
        if not os.path.exists(Config.TRIPLETS_FILE):
            logger.info("Triplets file not found. Generating triplets.")
            generate_and_save_triplets(Config.CSV_FILE, Config.TRIPLETS_FILE)
        
        logger.info("Tokenizing and saving triplets.")
        tokenize_and_save_triplets(Config.TRIPLETS_FILE, Config.TOKENIZED_TRIPLETS_FILE)
    else:
        logger.info("Tokenized triplets found. Skipping data preparation.")


def main():
    logger.info("Starting the Literary Style Analysis program")

    try:
        prepare_data()

        logger.info("Initializing dataset")
        start_time = time.time()
        dataset = LiteraryStyleDataset(Config.TOKENIZED_TRIPLETS_FILE, tokenized=True)
        initialization_time = time.time() - start_time
        logger.info(f"Dataset initialization completed in {initialization_time:.2f} seconds")

        logger.info("Preparing dataset")
        start_time = time.time()
        train_data, test_data = dataset.prepare_dataset()
        preparation_time = time.time() - start_time
        logger.info(f"Dataset preparation completed in {preparation_time:.2f} seconds")
        
        logger.info(f"Number of training triplets: {len(train_data)}")
        logger.info(f"Number of test triplets: {len(test_data)}")

        logger.info("Creating data loaders")
        train_loader = create_data_loader(train_data, Config.BATCH_SIZE)
        test_loader = create_data_loader(test_data, Config.BATCH_SIZE, shuffle=False)
        
        logger.info("Initializing model")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TripletLiteraryStyleNetwork().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

        start_epoch, _ = load_model(model, optimizer, get_model_path())

        logger.info("Starting model training")
        train_triplet(model, train_loader, optimizer, Config.NUM_EPOCHS - start_epoch, device)
        logger.info("Model training completed")

        logger.info("Starting model testing")
        test_loss, test_accuracy = test_triplet(model, test_loader, device)
        logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        logger.info("Displaying random triplet analysis")
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        display_random_triplet_similarities(model, dataset, tokenizer, device)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.exception("Stack trace:")

if __name__ == "__main__":
    main()