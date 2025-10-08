# Libraries
import os
import re
import math
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm 
from sklearn.metrics import classification_report
from datetime import datetime, timedelta
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import Counter
from scipy.stats import mode
import calendar

def process_day(date, signal_directory, search_window_size, stride):
    """
    Processes a daily GOES signal file into normalized sliding windows.

    Parameters:
        date (datetime.date): Date of the signal file to process.
        signal_directory (str): Directory containing the signal CSV files.
        search_window_size (int): Length of each sliding window.
        stride (int): Step size between consecutive windows.

    Returns:
        tuple: Three lists containing:
            - all_X (list): Normalized signal windows.
            - all_y (list): Empty placeholder for labels (reserved for compatibility).
            - timestamps (list): Timestamps corresponding to each window start.
    """

    # Construct file path for the day's signal
    fname = f"GOES_Signal_{date:%Y%m%d}.csv"
    fpath = os.path.join(signal_directory, fname)
    
    try:
         # Load the signal data
        df_goes_signal = pd.read_csv(fpath)
        S = df_goes_signal['Long_Channel_Flux']
        bt_ = df_goes_signal['time']
        
        # Skip processing if signal or timestamps are empty
        if S.empty or bt_.empty:
            return [], [], []
        
        all_X, all_y, timestamps = [], [], 
        
        # Generate sliding windows over the signal
        for j in range(0, len(S) - search_window_size, stride):
            window = S[j:j + search_window_size]

            # normalize input
            window = (window - np.mean(window)) / (np.std(window) + 1e-30)

            all_X.append(window)
            timestamps.append(bt_[j])
            
        return all_X, all_y, timestamps

    except Exception as e:
        # Handle file or processing errors
        print(f"[{date}] Error occurred: {e}")
        return [], [], []


def safe_process_day(args):
    """
    Safely executes process_day within a parallel worker, handling exceptions.

    Parameters:
        args (tuple): Contains (date, signal_directory, search_window_size, stride).

    Returns:
        tuple: Output from process_day or empty lists if an exception occurs.
    """
    
    date, signal_directory, search_window_size, stride = args
    
    try:
        # Execute main processing routine
        return process_day(date, signal_directory, search_window_size, stride)
    except Exception as e:
        # Log and suppress worker-level exceptions
        print(f"[{date}] Exception in worker: {e}")
        return [], [], []


# In[17]:


def preprocess_all_and_save_parallel(start_date, end_date, signal_directory, search_window_size, stride, output_file, max_workers):
    """
    Preprocesses GOES signal data across a date range in parallel and saves the results.

    Parameters:
        start_date (datetime.date): Starting date for processing.
        end_date (datetime.date): Ending date for processing.
        signal_directory (str): Directory containing signal CSV files.
        search_window_size (int): Length of each sliding signal window.
        stride (int): Step size between consecutive windows.
        output_file (str): Path to save the processed .npz file.
        max_workers (int): Number of parallel worker processes.

    Returns:
        str: Path to the saved preprocessed output file.
    """
    # Ensure integer window and stride values
    search_window_size = int(search_window_size)
    stride = int(stride)

    # Generate a list of all dates in the range
    date_list = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

    # Prepare argument tuples for each date
    args_list = [(date, signal_directory, search_window_size, stride) for date in date_list]

    all_X, all_y = [], []
    all_timestamps = []

    # Execute parallel processing across all dates
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for X_day, y_day, t_day in tqdm(executor.map(safe_process_day, args_list), total=len(args_list)):
            all_X.extend(X_day)
            all_timestamps.extend(t_day)

    # Convert processed results to NumPy arrays
    all_X = np.array(all_X, dtype=np.float32)

    # Save results to compressed file
    print(f"Total samples: {len(all_X)} | Saving to {output_file} ...")
    np.savez_compressed(output_file, X=all_X, y=all_y, timestamps=np.array(all_timestamps, dtype='datetime64[m]'))
    print("Done saving preprocessed data in parallel.")

    return output_file


# main
import time
start_time = time.time()

# Directory containing merged GOES signal files
signal_directory = 'directory to files/Merged_Signals_2018_2025'

search_w_size = 600  # Sliding window size
frac = 0.2           # Fraction for stride calculation

# --- Generate prediction datasets ---    
if __name__ == "__main__":
    from datetime import date
    
    preprocess_all_and_save_parallel(
        start_date = date(2024, 3, 18),
        end_date = date(2024, 2, 27),
        signal_directory = signal_directory,
        search_window_size = search_w_size,
        stride = int(search_w_size * frac),
        output_file = "directory to files/Prediction_set_2024.npz",
        max_workers=20      
    ) 
    

# --- Execution summary ---
end_time = time.time()
elapsed = end_time - start_time

print(f"Execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")




