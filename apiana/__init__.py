import logging
from rich.logging import RichHandler
from os import getenv

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

if getenv("APIANA_AI_DEBUG") is not None:
    import pdb

    pdb.set_trace()
