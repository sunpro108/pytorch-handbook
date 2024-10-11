import logging
import logging.config
import torch
from accelerate.logging import get_logger
from accelerate import Accelerator
print('test....')

logging.config.fileConfig('logging.ini', disable_existing_loggers=False)
logger = get_logger(__name__)


accelerator = Accelerator()
accelerator.print('......start test......'*2+'\n')
logger.info("My log, all process", main_process_only=False)
logger.debug("My log, main process", main_process_only=True)

accelerator.print('......test second log......'*2+'\n')
logger = get_logger(__name__, log_level="DEBUG")
logger.info("My log, default process only", main_process_only=True)
logger.debug("My second log")

accelerator.print('......test third log......'*2+'\n')
array = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
letter_at_rank = array[accelerator.process_index]
logger.info(letter_at_rank, in_order=True)
accelerator.print('......test end......'*2+'\n')
accelerator.end_training()
torch.distributed.destroy_process_group()