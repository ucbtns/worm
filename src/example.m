

clc; clear 

% Adding path directory
addpath ~/spm/toolbox/DEM/
addpath ~/spm/
addpath ~/sound_files
%--------------------------------------------------------------------------

% Call the model: 
mdp = model('default'); 
[mdp.MDP(1,1:4)] = deal(mdp.MDP);


clear global VOX TRAIN
rng('default')

global TRAIN
TRAIN =0;


% Train prosody: 
if TRAIN 
    % articulate questions and answers for 32 trials
    %----------------------------------------------------------------------
    VOX   = [0 0 0];
    M     = mdp;
    for t = 1:M.T
        M.MDP(t).VOX = VOX(t);
    end
    
    % remove preferences and create an array MDP structures
    %----------------------------------------------------------------------
    clear MDP
    M.C   = spm_zeros(M.C);
    [MDP(1,1)] = deal(M);
    MDP   = spm_MDP_VB_X_latest(MDP);
    
    % retrieve accumulated Dirichlet concentration parameters
    %----------------------------------------------------------------------
    da    = spm_unvec(spm_vec(MDP(end).mdp(end).a) - spm_vec(a),a);
    
    % supplement (Dirichlet prior) mapping
    %----------------------------------------------------------------------
    A     = mdp.MDP.A;                           % original likelihood
    a{1}  = spm_norm_a(A{1});                    % lexical outcomes
    for g = 2:numel(A)                           % prosody outcomes
        a{g} = spm_norm_a(a{g}) + spm_norm_a(da{g});
    end
    
    % save updated Dirichlet parameters and place illustrate prosody
    %======================================================================
    DIR   = fileparts(which('spm_voice.m'));
    save(fullfile(DIR,'RECORDING_DEMO'),'a')

end


% Example runs with WORM agents
%==========================================================================

 VOX   = [1,0,1];
for i = 1:20
    M     = mdp;
    for t = 1:M.T
        M.MDP(t).VOX = VOX(t);
    end
    clear MDP
    [MDP(1,1:20)] = deal(M);
    MDP  = spm_MDP_VB_X(MDP);
    save(strcat('~\control_', num2str(i),'.mat'), 'MDP');
    clear MDP
end

% 
%% Example runs with human: agent answers: [0 2 1 2] and is given feedback
% ==========================================================================

% Ask (A):
% ====================
VOX   = [2,1,1];
for i = 1:20
    M     = mdp;
    for t = 1:M.T
        M.MDP(t).VOX = VOX(t);
    end
    clear MDP
    [MDP(1,1:2)] = deal(M);
    MDP  = spm_MDP_VB_X(MDP);
    save(strcat('~\A_', num2str(i),'.mat'), 'MDP');
    clear MDP
end


% Repeat (B)
% ===================
VOX   = [1,2,1];
for i = 1:20
    M     = mdp;
    for t = 1:M.T
        M.MDP(t).VOX = VOX(t);
    end
    clear MDP
    [MDP(1,1:1)] = deal(M);
    MDP  = spm_MDP_VB_X_latest(MDP);
    save(strcat('~\B_', num2str(i),'.mat'), 'MDP');
    clear MDP
end

% Feedback (C)
% ==============================
VOX   = [1,1,2];
for i = 1:20
    M     = mdp;
    for t = 1:M.T
        M.MDP(t).VOX = VOX(t);
    end
    clear MDP
    [MDP(1,1:2)] = deal(M);
    MDP  = spm_MDP_VB_X(MDP);
    save(strcat('~\C_', num2str(i),'.mat'), 'MDP');
    clear MDP
end




