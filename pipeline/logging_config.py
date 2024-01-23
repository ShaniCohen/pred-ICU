import os
from datetime import datetime
import logging


def setup_logging():
    log_directory = os.path.abspath('../logs')
    os.makedirs(os.path.abspath(log_directory), exist_ok=True)

    log_filename = datetime.now().strftime("%Y-%m-%d.log")
    log_filepath = os.path.join(log_directory, log_filename)

    logging.basicConfig(
        filename=log_filepath,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
