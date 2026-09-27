import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Creates a structured logger with a standardized format.
    Format: [Timestamp] [Level] [Module]: Message
    """
    logger = logging.getLogger(name)
    
    # If the logger already has handlers, it's initialized. Don't add more.
    # (Prevents duplicate logs like "Info... Info...")
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Prevent logs from propagating to the root logger (avoids double printing)
    logger.propagate = False
    
    return logger