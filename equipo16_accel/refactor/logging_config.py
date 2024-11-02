# logging_config.py

import logging
import sys

def setup_logging(log_file: str = "pipeline.log") -> logging.Logger:
    """Sets up logging configuration.

    Args:
        log_file (str): Filename for the log file.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Add handlers
    logger.handlers = []
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
