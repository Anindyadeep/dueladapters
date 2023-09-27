import os 
import logging
from datetime import datetime
from duel_adapters import GeneralConfig


log_dir = GeneralConfig().log_dir
os.makedirs(log_dir, exist_ok=True)

# get the current dir and make the new file path for the app logs 

current_date = datetime.now().strftime("%Y-%m-%d")
logfile_path = os.path.join(
    log_dir,
    f"app_{current_date}.log"
)

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[
        logging.FileHandler(logfile_path),  
        logging.StreamHandler(),  
    ]
)

# make a logger instance 
logger = logging.getLogger("covergpt_logger")