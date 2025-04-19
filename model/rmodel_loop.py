

# 3-layer; with feedback; 2AFC

import subprocess
import os

output_dir = "/mnt/neurocube/local/serenceslab/holly/RNN_Geo/model"

for i in range(1, 2):  # Loop from 1 to 3 (inclusive)
    command = [
        "python",
        "main_feedback_v2.py",
        "--gpu", "0",
        "--gpu_frac", "0.70",
        "--n_trials", "20000",
        "--mode", "train",
        "--N1", "200",
        "--N2", "200",
        "--N3", "200",
        "--P_inh", "0.20",
        "--som_N", "0",
        "--task", "rdk_70_30",
        "--gain", "1.5",
        "--P_rec", "0.20",
        "--act", "sigmoid",
        "--loss_fn", "l2",
        "--apply_dale", "True",
        "--decay_taus", "4", "25",
        "--output_dir", output_dir,
    ]

    try:
        print(f"Starting iteration {i}")
        subprocess.run(command, check=True) #runs the command, raises an error if the return code is non zero
        print(f"Finished iteration {i}")

    except subprocess.CalledProcessError as e:
        print(f"Error during iteration {i}: {e}")
        print(f"command was: {command}")
        #You may want to exit here, or handle the error in a different way.
    except FileNotFoundError:
        print(f"Error: python script 'main_feedback_v2.py' not found.")
        break # or exit()

    print("Loop completed.")