"""
DemandCast AI - Enterprise Structured Logging Module
================================================================================
Configures centralized logging with console output and auto-rotating log files.
Provides standardized formatting, ISO-8601 timestamps, and log level controls.

Author: DemandCast AI Core Architecture Team
Assessment Phase: Stage 16 - Production Requirements
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone

LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
os.makedirs(LOGS_DIR, exist_ok=True)

DEFAULT_LOG_FILE = os.path.join(LOGS_DIR, "demandcast_production.log")


def get_logger(name: str = "DemandCastAI", log_file: str = DEFAULT_LOG_FILE, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured thread-safe logger instance with both console streaming
    and size-capped rotating file persistence.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if already configured
    if logger.handlers:
        return logger

    # 1. Custom Standard Formatter
    log_format = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 2. Console Stream Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # 3. Rotating File Handler (Max 10MB per file, up to 5 backup logs)
    try:
        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Failed to initialize rotating file handler at {log_file}: {e}")

    return logger


# Global default logger instance
logger = get_logger()

if __name__ == "__main__":
    logger.info("DemandCast AI Structured Logging Module Initialized Successfully.")
    logger.warning("Testing warning alert level.")
    logger.error("Testing error alert level.")
