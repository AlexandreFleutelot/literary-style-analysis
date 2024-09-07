import logging
import os
from logging.handlers import RotatingFileHandler
from src.utils.config import Config

def setup_logger():
    """
    Function to setup a single logger for the entire application
    """
    logger = logging.getLogger('literary_style_analysis')
    
    # Clear any existing handlers to avoid duplicate logs
    if logger.handlers:
        logger.handlers.clear()
    
    # Set log level from Config
    log_level = getattr(logging, Config.LOG_LEVEL.upper())
    logger.setLevel(log_level)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create logs directory if it doesn't exist
    if not os.path.exists(Config.LOG_DIR):
        os.makedirs(Config.LOG_DIR)
    
    # File Handler
    file_handler = RotatingFileHandler(
        os.path.join(Config.LOG_DIR, 'literary_style_analysis.log'),
        maxBytes=5*1024*1024,  # 5 MB
        backupCount=3
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Create a single logger instance
logger = setup_logger()