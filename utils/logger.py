import logging
from typing import Dict

# Initialize loggers dictionary to keep track of created loggers
loggers: Dict[str, logging.Logger] = {}

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger by name."""
    # If logger already exists, return it
    if name in loggers:
        return loggers[name]
    
    # Create and configure a new logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set formatter for the handler
    console_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(console_handler)
    
    # Store logger in dictionary
    loggers[name] = logger
    
    return logger