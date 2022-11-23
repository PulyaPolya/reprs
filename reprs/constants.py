import os

UNKNOWN_TOKEN = os.environ.get("UNKNOWN_TOKEN", "<UNK>")
PAD_TOKEN = os.environ.get("PAD_TOKEN", "<PAD>")
START_TOKEN = os.environ.get("START_TOKEN", "<START>")
END_TOKEN = STOP_TOKEN = os.environ.get("STOP_TOKEN", "<STOP>")
SPECIALS = (PAD_TOKEN, UNKNOWN_TOKEN, START_TOKEN, STOP_TOKEN)
