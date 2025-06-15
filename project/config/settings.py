import os
from dotenv import load_dotenv

load_dotenv()

TIME_GPT_API_KEY = os.getenv("TIME_GPT_API_KEY", "")
