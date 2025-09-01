
import os
from dotenv import load_dotenv

load_dotenv()  # load .env if present

DATA_PATH = os.getenv("DATA_PATH", os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed_data.csv"))
TOP_N_TOPIC_PREVIEW = int(os.getenv("TOP_N_TOPIC_PREVIEW", "5"))
