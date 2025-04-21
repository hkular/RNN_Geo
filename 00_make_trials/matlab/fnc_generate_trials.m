% Name: Robert Kim
% Date: 03-29-2023
% Email: robert.f.kim@gmail.com
% fnc_generate_trials.m
% Description: Function to generate trials for a task
% HK added random seed

function [u, lab] = fnc_generate_trials(task, task_info)
% INPUT
%   task: task name ('instr', 'xor', etc...)
%   task_info: struct containing task information (stim onset time, 
%   stim duration, delay duration, etc...)
% OUTPUT
%   u: input matrix
%   lab: struct containing trial info (instruction timing/amp and stim info)


% 80-20 split with low coherence
if strcmpi(task, 'rdk')
  trial_dur = task_info.trial_dur;
  stim_on = task_info.stim_on;
  stim_dur = task_info.stim_dur;
  num_stims = task_info.num_stims;
  predominant_stim = task_info.pred;
  coh = task_info.coh; % coherence level

  % probability for the primary stim (first stim)
  % Ex. primary_prob = 0.80 means 80% for the first stim and the rest are equally distributed
  primary_prob = task_info.primary_prob;

  u = zeros(num_stims, trial_dur);
  u = u + randn(size(u)); % add noise

  rand_num = rand;

  if rand_num < primary_prob
    u(predominant_stim, stim_on:stim_on+stim_dur) = u(predominant_stim, stim_on:stim_on+stim_dur) + coh;
    lab = predominant_stim;
  else
    rand_choice = randperm(num_stims-1);
    choices = setdiff([1:num_stims], predominant_stim);
    rand_choice = choices(rand_choice(1));
    u(rand_choice, stim_on:stim_on+stim_dur) = u(rand_choice, stim_on:stim_on+stim_dur) + coh;
    lab = rand_choice;
  end

elseif strcmpi(task, 'instr')
  trial_dur = task_info.trial_dur;
  stim_on = task_info.stim_on;
  stim_dur = task_info.stim_dur;
  delay = task_info.delay;

  if ~isempty(task_info.force) % Generate specific trial instead of random trial
  % task_info.force for `instr` is [stim1, stim2, instr_amp, instr_t]
    u = zeros(3, trial_dur);

    if task_info.force(end) == -1
      instr_on = stim_on - 50;
    else
      instr_on = stim_on + stim_dur;
    end
    instr_amp = task_info.force(3);

    u(end, instr_on:instr_on+50) = instr_amp;
    u(1, stim_on:stim_on+stim_dur) = task_info.force(1);
    u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = task_info.force(2);

    lab.stim_lab = prod(task_info.force(1:2));
    lab.instr_t = task_info.force(end);
    lab.instr_amp = instr_amp;
    lab.stim_ids = task_info.force(1:2);

  elseif isempty(task_info.force)
    u = zeros(3, trial_dur);

    % First set the instruciton signal onset and amplitude
    % Instruction timing (either before or after first stim)
    % instr_t indicates the timing (-1 = before the first stim, +1 = after the first stim)
    if rand < 0.50
      instr_on = stim_on - 50;
      instr_t = -1;
    else
      instr_on = stim_on + stim_dur;
      instr_t = 1;
    end
    
    % Instruction amplitude (+1 for DMS and -1 for anti-DMS)
    if rand < 0.50
      instr_amp = 1; % DMS task
    else
      instr_amp = -1; % anti-DMS task
    end
    %u(end, instr_on:stim_on+stim_dur*2+delay) = instr_amp;
    u(end, instr_on:instr_on+50) = instr_amp;

    % Now set the first and second stimuli
    stim_labs = zeros(1, 2);
    if rand < 0.5
      u(1, stim_on:stim_on+stim_dur) = 1;
      stim_labs(1, 1) = 1;
    else
      u(1, stim_on:stim_on+stim_dur) = -1;
      stim_labs(1, 1) = -1;
    end

    if rand < 0.5
      u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = 1;
      stim_labs(1, 2) = 1;
    else
      u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = -1;
      stim_labs(1, 2) = -1;
    end
    lab.stim_lab = prod(stim_labs);
    lab.instr_t = instr_t;
    lab.instr_amp = instr_amp;
    lab.stim_ids = stim_labs;
  end

elseif strcmpi(task, 'instr2')
  trial_dur = task_info.trial_dur;
  stim_on = task_info.stim_on;
  stim_dur = task_info.stim_dur;
  delay = task_info.delay;

  u = zeros(6, trial_dur);

  % First set the instruciton signal onset and amplitude
  % Instruction timing (either before or after first stim)
  % instr_t indicates the timing (-1 = before the first stim, +1 = after the first stim)
  if rand < 0.50
    instr_on = stim_on - 50;
    instr_t = -1;
  else
    instr_on = stim_on + stim_dur;
    instr_t = 1;
  end
  
  % Instruction amplitude (+1 for DMS and -1 for anti-DMS)
  if rand < 0.50
    instr_amp = 1; % DMS task
  else
    instr_amp = -1; % anti-DMS task
  end
  %u(end, instr_on:stim_on+stim_dur*2+delay) = instr_amp;
  u(4, instr_on:instr_on+51) = instr_amp;

  % Cue signal (-1 for the first modality and +1 for the second modality)
  if rand < 0.50
    cue_amp = -1; % focus on the first modality
  else
    cue_amp = 1; % focus on the second modality
  end
  u(6, instr_on:instr_on+50) = cue_amp;

  % Set the first and second stimuli for the first modality
  stim_labs1 = zeros(1, 2);
  if rand < 0.5
    u(1, stim_on:stim_on+stim_dur) = 1;
    stim_labs1(1, 1) = 1;
  else
    u(1, stim_on:stim_on+stim_dur) = -1;
    stim_labs1(1, 1) = -1;
  end

  if rand < 0.5
    u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = 1;
    stim_labs1(1, 2) = 1;
  else
    u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = -1;
    stim_labs1(1, 2) = -1;
  end

  % Set the first and second stimuli for the second modality
  stim_labs2 = zeros(1, 2);
  if rand < 0.5
    u(3, stim_on:stim_on+stim_dur) = 1;
    stim_labs2(1, 1) = 1;
  else
    u(3, stim_on:stim_on+stim_dur) = -1;
    stim_labs2(1, 1) = -1;
  end

  if rand < 0.5
    u(4, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = 1;
    stim_labs2(1, 2) = 1;
  else
    u(4, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = -1;
    stim_labs2(1, 2) = -1;
  end
  lab.stim_mod1 = stim_labs1;
  lab.stim_mod2 = stim_labs2;
  lab.stim_lab1 = prod(stim_labs1);
  lab.stim_lab2 = prod(stim_labs2);

  lab.instr_t = instr_t;
  lab.instr_amp = instr_amp;
  lab.cue_amp = cue_amp;
end




