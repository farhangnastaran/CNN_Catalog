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
from datetime import datetime, timedelta, date
from datetime import date
from collections import Counter
from scipy.stats import mode
import random
import calendar

def CSV_Manually_Selected_INFO(Event_df, target_date):
    """
    Extracts event start and peak times from a DataFrame for a specific date.

    Parameters:
        Event_df (pd.DataFrame): DataFrame containing event details with 'date', 'start time', and 'peak time' columns.
        target_date (datetime.date): Target date to filter events.

    Returns:
        tuple: Two lists containing datetime objects for event start times and peak times on the target date.
    """
    # Ensure 'date' column is in datetime format
    Event_df['date'] = pd.to_datetime(Event_df['date'], errors='coerce')
    
    # Filter rows matching the target date
    day_events = Event_df[Event_df['date'].dt.date == target_date].copy()

    # Combine date with start and peak times, converting to datetime objects
    ev_start = pd.to_datetime(
        day_events['date'].dt.strftime("%Y-%m-%d") + " " + day_events['start time'].astype(str),
        errors="coerce"
    ).tolist()

    ev_peak = pd.to_datetime(
        day_events['date'].dt.strftime("%Y-%m-%d") + " " + day_events['peak time'].astype(str),
        errors="coerce"
    ).tolist()

    return ev_start, ev_peak



def label_signal(t, ev_start, ev_peak):
    """
    Generates binary labels for time points based on event start and peak times.

    Parameters:
        t (array-like): Sequence of time points.
        ev_start (list): List of event start datetime objects.
        ev_peak (list): List of event peak datetime objects.

    Returns:
        np.ndarray: Array of integer labels (1 during event period, 0 otherwise).
    """
    # Initialize label array with zeros
    labels = np.zeros_like(t, dtype=int)

    # Convert time sequence to datetime format
    t = pd.to_datetime(t)

    # Label intervals between event start and peak times
    for s_time, p_time in zip(ev_start, ev_peak):
        if s_time is pd.NaT or p_time is pd.NaT:
            continue
        # Identify points within the event window
        mask = (t >= s_time) & (t <= p_time)
        labels[mask] = 1

    return labels


def process_day(date, signal_directory, search_window_size, stride, n_layer):
    """
    Processes a daily GOES signal file, generating normalized signal windows and labels.

    Parameters:
        date (datetime.date): Date of the signal file to process.
        signal_directory (str): Directory path containing signal CSV files.
        search_window_size (int): Size of each sliding window.
        stride (int): Step size between consecutive windows.
        n_layer (int): Model layer parameter (unused in this function but retained for consistency).

    Returns:
        tuple: Three lists containing signal windows (all_X), label windows (all_y), 
               and corresponding timestamps.
    """
    global fav_inst_df

    # Construct file path for the day's signal
    fname = f"GOES_Signal_{date:%Y%m%d}.csv"
    fpath = os.path.join(signal_directory, fname)

    try:
        # Load signal data
        df_goes_signal = pd.read_csv(fpath)
        S = df_goes_signal['Long_Channel_Flux']
        bt_ = df_goes_signal['time']

        # Skip empty files
        if S.empty or bt_.empty:
            return [], [], []

        # Retrieve event start and peak times for the day
        ev_start, ev_peak = CSV_Manually_Selected_INFO(fav_inst_df, date)
        
        # Generate binary labels for the signal
        labels = label_signal(bt_, ev_start, ev_peak)

        all_X, all_y, timestamps = [], [], []

        # Slide window across the signal
        for j in range(0, len(S) - search_window_size, stride):
            window = S[j:j + search_window_size]
            label_window = labels[j:j + search_window_size]            

            # Normalize signal window
            window = (window - np.mean(window)) / (np.std(window) + 1e-30)

            all_X.append(window)
            all_y.append(label_window)
            timestamps.append(bt_[j])

        return all_X, all_y, timestamps

    except Exception as e:
        # Log error and return empty outputs
        print(f"[{date}] Error occurred: {e}")
        return [], [], []


def safe_process_day(args):
    """
    Wrapper for process_day to ensure safe parallel execution with error handling.

    Parameters:
        args (tuple): Contains (date, signal_directory, search_window_size, stride, n_layer).

    Returns:
        tuple: Output from process_day or empty lists if an exception occurs.
    """
    date, signal_directory, search_window_size, stride, n_layer = args
    global fav_inst_df

    try:
        # Execute main processing routine
        return process_day(date, signal_directory, search_window_size, stride, n_layer)
    except Exception as e:
        # Capture and log worker-level exceptions
        print(f"[{date}] Exception in worker: {e}")
        return [], [], []



def preprocess_all_and_save_parallel_train(
    list_dates, signal_directory, search_window_size, stride, n_layer, output_file, max_workers
):
    """
    Preprocesses GOES signal data across multiple dates in parallel and saves results.

    Parameters:
        list_dates (list): List of dates to process.
        signal_directory (str): Directory path containing signal CSV files.
        search_window_size (int): Size of each sliding signal window.
        stride (int): Step size between consecutive windows.
        n_layer (int): Model layer parameter (reserved for compatibility).
        output_file (str): Path for the output .npz file.
        max_workers (int): Number of parallel worker processes.

    Returns:
        str: Path to the saved preprocessed output file.
    """
    global fav_inst_df

    # Ensure integer parameters
    search_window_size = int(search_window_size)
    stride = int(stride)

    # Prepare argument tuples for parallel processing
    args_list = [(date, signal_directory, search_window_size, stride, n_layer) for date in list_dates]

    all_X, all_y, all_timestamps = [], [], []

    # Execute parallel preprocessing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for X_day, y_day, t_day in tqdm(executor.map(safe_process_day, args_list), total=len(args_list)):
            all_X.extend(X_day)
            all_y.extend(y_day)
            all_timestamps.extend(t_day)

    # Convert to NumPy arrays
    all_X = np.array(all_X, dtype=np.float32)
    all_y = np.array(all_y, dtype=np.uint8)

    # Save processed data
    print(f"Total samples: {len(all_X)} | Saving to {output_file} ...")
    np.savez_compressed(
        output_file,
        X=all_X,
        y=all_y,
        timestamps=np.array(all_timestamps, dtype='datetime64[m]')
    )
    print("Done saving preprocessed data in parallel.")

    return output_file


def preprocess_all_and_save_parallel_test(
    list_dates, signal_directory, search_window_size, stride, n_layer, output_file, max_workers
):
    """
    Preprocesses GOES signal data for testing across multiple dates in parallel and saves results.

    Parameters:
        list_dates (list): List of dates to process.
        signal_directory (str): Directory containing signal CSV files.
        search_window_size (int): Size of each sliding window over the signal.
        stride (int): Step size between consecutive windows.
        n_layer (int): Model layer parameter (retained for consistency).
        output_file (str): Destination path for the output .npz file.
        max_workers (int): Number of worker processes for parallel execution.

    Returns:
        str: Path to the saved preprocessed test data file.
    """
    global fav_inst_df

    # Ensure numeric parameters
    search_window_size = int(search_window_size)
    stride = int(stride)

    # Build argument list for parallel workers
    args_list = [(date, signal_directory, search_window_size, stride, n_layer) for date in list_dates]

    all_X, all_y, all_timestamps = [], [], []

    # Parallel processing using multiple workers
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for X_day, y_day, t_day in tqdm(executor.map(safe_process_day, args_list), total=len(args_list)):
            all_X.extend(X_day)
            all_y.extend(y_day)
            all_timestamps.extend(t_day)

    # Convert accumulated data to NumPy arrays
    all_X = np.array(all_X, dtype=np.float32)
    all_y = np.array(all_y, dtype=np.uint8)

    # Save preprocessed test data
    print(f"Total samples: {len(all_X)} | Saving to {output_file} ...")
    np.savez_compressed(
        output_file,
        X=all_X,
        y=all_y,
        timestamps=np.array(all_timestamps, dtype='datetime64[m]')
    )
    print("Done saving preprocessed data in parallel.")

    return output_file




# --- Main Execution Script ---

import time
start_time = time.time()

# Directory containing merged GOES signal files
signal_directory = 'directory/Merged_Signals_2018_2025'

# Load selected event metadata
Event_df = pd.read_csv("directory/selected_events_filtered.csv", parse_dates=["date"])

# Set global reference DataFrame
global fav_inst_df
fav_inst_df = Event_df

# --- Define training and validation date ranges ---

# --- training dates ---
train_mask = (fav_inst_df['date'] >= "2018-01-01") & (fav_inst_df['date'] < "2024-05-10")
list_dates_train = sorted(fav_inst_df.loc[train_mask, 'date'].dt.date.unique())

# --- validation dates ---
val_mask = (fav_inst_df['date'] >= "2024-05-10") & (fav_inst_df['date'] <= "2024-05-31")
list_dates_val = sorted(fav_inst_df.loc[val_mask, 'date'].dt.date.unique())

# -------------------------------------

search_w_size = 600  # Sliding window size
frac = 0.2           # Fraction for stride calculation

# --- Train set preprocessing ---
if __name__ == "__main__":
    from datetime import date
  
    preprocess_all_and_save_parallel_train(
        list_dates_train,
        signal_directory = signal_directory,
        search_window_size = search_w_size,
        stride = int(search_w_size * frac),
        n_layer = 4,
        output_file = "directory/Train_set_manual.npz",
        max_workers=20      
    )

# --- Validation set preprocessing ---
if __name__ == "__main__":
    from datetime import date
    
    preprocess_all_and_save_parallel_test(
        list_dates_val,
        signal_directory = signal_directory,
        search_window_size = search_w_size,
        stride = int(search_w_size * frac),
        n_layer = 4,
        output_file = "directory/Test_set_manual.npz",
        max_workers=20      
    )


# --- Validation set preprocessing ---
end_time = time.time()
elapsed = end_time - start_time

print(f"Execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")





