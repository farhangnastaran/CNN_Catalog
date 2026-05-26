# Solar_Flar_CNN_Catalog

A CNN-Derived Catalog for Solar Flares from GOES SXR Observations

This repository contains scripts for preprocessing and labeling GOES (Geostationary Operational Environmental Satellites) SXR flux data, generating training, validation, and prediction datasets for solar flare identification.


# Repository Structure

CNN_SXR_Generate_Samples_Train_Manual_Set.py                   --->   Preprocesses GOES signal data with manually selected flare events to generate labeled training and validation datasets.

CNN_SXR_Generate_Samples_Prediction.py                         --->   Preprocesses raw GOES signals to generate prediction datasets (unlabeled windows) across defined years.

CNN_SXR_Generate_Samples_Terminal_Execution.ipynb	             --->   Jupyter version of the prediction generator for interactive execution and debugging.

Verify_Label_Alignment_Training_Test_Sets_Visualization.ipynb	 --->   Utility notebook for visual inspection and verification of labeled datasets’ temporal alignment.

CNN_Catalog_Manual_4layer_BiLSTM_Trans.ipynb	                 --->   CNN architecture for flare identification.


# How to Run

The user should first execute "CNN_SXR_Generate_Samples_Terminal_Execution.ipynb" to generate the training, validation, and prediction datasets.
The training and validation datasets are constructed based on information recorded in the reference catalog, which includes approximately 7,700 events.
A selection of GOES SXR signals in CSV format is available in the Merged_Signals_2018_2025 folder and can be used for prediction purposes.

Once the datasets are generated, the user has two options:
  1) Train the model — Run "CNN_Catalog_Manual_4layer_BiLSTM_Trans.ipynb" to build the CNN model and then execute it for prediction.
  2) Use the pre-trained model — Load the existing model file "CNN_Model_Manual_4layer_BiLSTM_Trans_Encoder.pt" and directly perform the prediction step.

In either case, the identified events from the prediction sets will be recorded in a CSV file, which includes start times and peak times of the detected events.

A final post-processing step using the "Catalog_Refinement_Plot.ipynb" is recommended. This step refines the raw event detections by ensuring that the start and peak times identified by the CNN align with the local minima and maxima of the signal. It also merges overlapping events occurring within the same rise phase. The script produces: a refined event catalog saved as a CSV file, and plots a signal along with the corresponding detected raw and refined events.


