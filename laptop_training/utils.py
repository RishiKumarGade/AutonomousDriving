import os
from datetime import datetime

def get_save_path():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, f"ppo_metadrive_{timestamp}")
