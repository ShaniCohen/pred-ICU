import logging

def setup_logging():
    logging.basicConfig(filename='C:\\Users\\nirro\Desktop\MSc\predictive_modeling_healthcare\git\pred-ICU\pipeline\project_log.log',
                        level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode='a',  # Append mode
                        datefmt='%Y-%m-%d %H:%M:%S',
                        )

