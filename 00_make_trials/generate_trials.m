% Name: Robert Kim
% Date: 06-09-2023
% Email: rkim@salk.edu
% generate_trials.m
% Description: Script to generate trials
% 2-12-2024 HK - changed data path

clear; clc;

% ***************************************************************************************
% 70/30 split for 6AFC with interlayer feedforward connections only. High coherence (0.7)
% ***************************************************************************************
% Data directory
data_dir = 'data/rdk_70_30_6afc/feedforward_only/hi_coh';

feedback = false; % interlayer feedback (true or false)

% Get all the trained models (should be 40 .mat files)
mat_files = dir(fullfile(data_dir, '*.mat'));

% Choose one model as an example
model_path = fullfile(data_dir, mat_files(1).name);

% **Testing task condition (70-30 6-AFC with high coherence)
task_info = struct();
task_info.trials = 100;
task_info.trial_dur = 250; % trial duration (timesteps)
task_info.stim_on = 80;
task_info.stim_dur = 50;
task_info.num_stims = 6; % 6AFC
task_info.pred = 1; % predominant stimulus is "1"
task_info.coh = 0.7; % hi_coh = 0.7 vs. lo_coh = 0.6
task_info.primary_prob = 0.70; % 70-30 split

% NOTE: adjust pred and primary_prob to change the testing environement
% Ex. Evenly distributed 6-AFC => task_info.primary_prob = 1/6;
% Ex. pred=5 and primary_prob = 0.7 => stim "5" will be predominant (70% of the time)

outs = zeros(task_info.trials, task_info.trial_dur);
labs = zeros(task_info.trials, 1);
for tri = 1:task_info.trials
  % Generate trials
  [u, lab] = fnc_generate_trials('rdk', task_info);

  % Now test the trained model
  out = fnc_eval_model(model_path, u, feedback);

  outs(tri, :) = out('O'); % Store the output signal
  labs(tri) = lab;
end

% ***************************************************************************************
% 80/20 split for 2AFC with interlayer feedforward AND feedback. Low coherence (0.6)
% ***************************************************************************************
% Data directory
data_dir = '/data/prob_rnn/models/rdk_80_20/P_rec_0.2_Taus_4.0_25.0/training_1_0/with_feedback/lo_coh';

feedback = true; % interlayer feedback (true or false)

% Get all the trained models (should be 40 .mat files)
mat_files = dir(fullfile(data_dir, '*.mat'));

% Choose one model as an example
model_path = fullfile(data_dir, mat_files(1).name);

feedback% **Testing task condition (80-20 2-AFC with low coherence)
task_info = struct();
task_info.trials = 100;
task_info.trial_dur = 250; % trial duration (timesteps)
task_info.stim_on = 80;
task_info.stim_dur = 50;
task_info.num_stims = 2; % 2AFC
task_info.pred = 1; % predominant stimulus is "1"
task_info.coh = 0.6; % hi_coh = 0.7 vs. lo_coh = 0.6
task_info.primary_prob = 0.80; % 80-20 split

outs = zeros(task_info.trials, task_info.trial_dur);
labs = zeros(task_info.trials, 1);
for tri = 1:task_info.trials
  % Generate trials
  [u, lab] = fnc_generate_trials('rdk', task_info);

  % Now test the trained model
  out = fnc_eval_model(model_path, u, feedback);

  outs(tri, :) = out('O'); % Store the output signal
  labs(tri) = lab;
end



