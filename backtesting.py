import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import logging
from bisect import bisect
from preprocessing import

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s',)

if __name__ == '__main__':
    # Config
    CORR_WINDOW = 252
    RESIDUAL_WINDOW = 60
    N_COMPONENTS = 15
    DATE_COL = 'DlyCalDt'
    RET_COL = 'DlyRet'
    ID_COL = 'PERMNO'
    load_dotenv(".env")




