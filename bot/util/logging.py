import logging, sys, os

def init_logging():
    # initialize root logger only once
    if getattr(init_logging, "_done", False):
        return
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    init_logging._done = True

def get_logger(name: str):
    return logging.getLogger(name)