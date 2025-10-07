# CNN_Catalog
A CNN-Derived Catalog of Solar Flares from GOES SXR Observations

This repository contains scripts for preprocessing and labeling GOES (Geostationary Operational Environmental Satellites) SXR flux data, generating training, validation, and prediction datasets for solar flare identification.

# Repository Structure
CNN_SXR_Generate_Samples_Train_Manual_Set.py                   --->   Preprocesses GOES signal data with manually selected flare events to generate labeled training and validation datasets.
CNN_SXR_Generate_Samples_Prediction.py                         --->   Preprocesses raw GOES signals to generate prediction datasets (unlabeled windows) across defined years.
CNN_SXR_Generate_Samples_Terminal_Execution.ipynb	             --->   Jupyter version of the prediction generator for interactive execution and debugging.
Verify_Label_Alignment_Training_Test_Sets_Visualization.ipynb	 --->   Utility notebook for visual inspection and verification of labeled datasets’ temporal alignment.
CNN_Catalog_Manual_4layer_BiLSTM_Trans.ipynb	                 --->   CNN architecture for flare identification.



