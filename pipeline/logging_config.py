import logging
import os


def setup_logging():
    logging.basicConfig(filename=os.path.abspath('.\\project_log.log'),
                        level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode='a',  # Append mode
                        datefmt='%Y-%m-%d %H:%M:%S',
                        )
