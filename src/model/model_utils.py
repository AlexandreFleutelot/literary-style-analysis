import os
import torch
from src.utils.config import Config
from src.utils.logger import logger

def save_model(model, optimizer, epoch, loss, path):
    """
    Save the model state, optimizer state, epoch, and loss.
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }, path)
    logger.info(f"Model saved to {path}")

def load_model(model, optimizer, path):
    """
    Load the model state, optimizer state, epoch, and loss.
    """
    if os.path.exists(path):
        # Add map_location to handle CPU/GPU transitions
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        logger.info(f"Model loaded from {path}")
        return epoch, loss
    else:
        logger.warning(f"No model found at {path}")
        return 0, None

def get_model_path(epoch=None):
    """
    Get the path for saving/loading the model.
    """
    if not os.path.exists(Config.MODEL_SAVE_DIR):
        os.makedirs(Config.MODEL_SAVE_DIR)
    
    if epoch is not None:
        return os.path.join(Config.MODEL_SAVE_DIR, f"model_epoch_{epoch}.pth")
    else:
        return os.path.join(Config.MODEL_SAVE_DIR, "model_final.pth")