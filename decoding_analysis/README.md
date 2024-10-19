Code to examine the pre-trained continuous-variable RNNs trained on a psychophysical task [Rungratsameetaweemana et al, PNAS 2019](https://www.pnas.org/doi/pdf/10.1073/pnas.1904502116)

00_generate_trials.py:  generates trials using the functions defined in fnc_eval_model.py and fnc_generate_trials.py

    fnc_generate_trials.py: generates trials for the task  

    fnc_eval_model.py: implementation of rate model

01_decode_pipeline.py: implements linear svm decoder loading trials and rnn firing rates saved from 00_generate_trials.py


data folder contains example npz files containing trial data for neutral and unexpected conditions