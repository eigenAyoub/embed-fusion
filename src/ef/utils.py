import os
import pandas as pd
from ef.config import *


def merge_index(index1, index2):
    index = index1.union(index2).unique()
    index = index.sort_values()
    return index


def update_meta(df: pd.DataFrame, filename: str):
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
    
    path = os.path.join(RESULTS_FOLDER, filename)

    # Check if the results file exists
    if os.path.exists(path):
        if isinstance(df.index, pd.MultiIndex):
            num_index_levels = len(df.index.levels)
        else:
            num_index_levels = 1
        df_loaded = pd.read_csv(path, index_col=list(range(num_index_levels)), header=0)
        index = merge_index(df.index, df_loaded.index)
        df_new = pd.DataFrame(df_loaded, index=index, columns=df.columns, dtype=object)
        df_new.update(df)
        df = df_new

    # Save the results to disk
    df.to_csv(path)