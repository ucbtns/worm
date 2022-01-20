function mdp = model(seed) 

% ===========================================================================            

rng(seed)
 
 
%% first level (Lexical - Articulatory Level)
%==========================================================================
% question 1: {'repeat'}
%--------------------------------------------------------------------------
% probabilistic mapping from hidden states to outcomes: A
%--------------------------------------------------------------------------
% Setup Lexicon (synonyms) and syntactic (sentence) structures
%==========================================================================
word{1}     = {'can'}; 
word{7}    = {'you'};
word{11} = {'repeat'};
word{6}    = {'please'};
word{2}     = {'yes'};
word{3}     = {'no'};
word{4}     = {'sure'};
word{5}    = {'OK'};
word{8}    = {'Iam'};
word{9} = {'ready'};
word{10} =  {'go'}; 
word{12} = { 'sorry'};
word{13}     = {'not'};

familiar   = {'square', 'green', 'blue', 'red','triangle'}; 
feedback = {'correct','wrong',' '};

% allowable outcomes (words)
%--------------------------------------------------------------------------
outcome     = unique([word{:}  familiar{:} feedback{:}]);

% allowable sentences: ending in the same (space) outcome: s10 = ' '
%==========================================================================

% allowable sentences  f1 = familiar, f3 = familiar, f2 = feedback
% syntax    = {Q1-F,answer,feedback,??};
%--------------------------------------------------------------------------
sentence{1} = {{'s5', 's1','s7', 's6', 's11',  'f1',}};
sentence{2} = {{'s5','f1'} {'s2','f1'} {'s4', 'f1'}};
sentence{3} = {{'f2'}};
sentence{4} = {{'s8','s13', 's9'} {'s8','s13','s4'}, {'s3', 's12'}};

% create semantic (lexical x syntax) labels
%--------------------------------------------------------------------------
for i = 1:numel(sentence)
    for j = 1:numel(sentence{i})
        for k = 1:numel(sentence{i}{j})
            sentence{i}{j}{k} = sprintf('s%-3.0d%s',i,sentence{i}{j}{k});
        end
    end
end
semantics = [sentence{:}];
semantics = unique([semantics{:}]);

% assemble hidden (syntax) states
%--------------------------------------------------------------------------
label.factor{1} = 'audition';   label.name{1}  = familiar;
label.factor{2} = 'feedback.';   label.name{2}  = feedback;
label.factor{3} = 'syntax'; label.name{3}  = semantics;

% prior beliefs about initial states D 
%--------------------------------------------------------------------------
for i = 1:numel(label.factor)
    n    = numel(label.name{i});
    D{i} = ones(n,1)/n;
end

% initial states: syntax = semantics(find(D{4})) :=syntax    = {Q1-F,Q2-F, go, reply,feedback,??};
%--------------------------------------------------------------------------
state = label.name{3};
D{3}  = spm_zeros(D{3});
for i = 1:numel(sentence)
    j       = ismember(state,sentence{i}{1}(1));
    D{3}(j) = 1;
end

% hidden factors
%--------------------------------------------------------------------------
Nf    = numel(D);
for f = 1:Nf
    Ns(f) = numel(D{f});
end

% single outcome modality with multiple phrases
%--------------------------------------------------------------------------
levels = {'1','2','3','4','5'};
label.modality{1} = 'word';       label.outcome{1} = outcome;
label.modality{2} = 'amplitude';  label.outcome{2} = levels;
label.modality{3} = 'duration';     label.outcome{3} = levels;
label.modality{4} = 'pitch';      label.outcome{4} = levels;
label.modality{5} = 'frequency';  label.outcome{5} = levels;
label.modality{6} = 'inflexion';  label.outcome{6} = levels;
label.modality{7} = 'biflexion';  label.outcome{7} = levels;

for f1 = 1:Ns(1) % Familiar Word; 
    for f2 = 1:Ns(2) % Feedback 
        for f3 = 1:Ns(3) % Syntax
    
                
                % indices of the state
                %----------------------------------------------------------
                j = {f1,f2,f3};
                
                % words under this state
                %==========================================================
                name = label.name{3}{f3}(5:end);
                if name(1) == 'f'
                    
                    % get repeat or feedback words:
                    %------------------------------------------------------
                    p   = eval(name(end));
                    q   = eval(name);
                    out = label.name{p}(q);
                    
                elseif name(1) == 's'
                    
                    % otherwise get synonyms
                    %------------------------------------------------------
                    q   = eval(name(2:end));
                    out = word{q};
                end
                
                % place words in likelihood matrix
                %----------------------------------------------------------
                i = ismember(outcome,out);
                A{1}(i,j{:}) = 1/sum(i);
                
                % prosody under this state
                %==========================================================
                for p = 2:numel(label.modality)
                    Ap           = [1 2 16 32 32 16 2 1];
                    A{p}(:,j{:}) = Ap(:)/sum(Ap);
                 end
        end
    end
end


% retrieve previously learned likelihood (Dirichlet) parameters
%==========================================================================
Ng  = numel(A);
try
    % load previous Dirichlet parameters (a)
    %----------------------------------------------------------------------
    load('D:\PhD\Code\spm\toolbox\DEM\RECORDING_DEMO.mat', 'a')
    % share prosody concentration parameters
    %----------------------------------------------------------------------
    a{1}  = spm_norm_a(A{1},16384);            % lexical parameters
    for g = 2:Ng                               % prosody parameters
        for i = 1:size(a{g},1)
            for j = 1:size(a{g},5)
                a{g}(i,:,:,:,j) = mean(mean(mean(a{g}(i,:,:,:,j),2),3),4);
            end
        end
    end    
    % normalise likelihood (Dirichlet prior) mapping (a)
    %----------------------------------------------------------------------
    for g = 1:Ng
        a{g} = spm_norm_a(a{g},16384);
    end
catch
    % create a new likelihood (Dirichlet prior) mapping (a)
    %----------------------------------------------------------------------
    for g = 1:Ng
        a{g} = spm_norm_a(A{g},16384);
    end
end

% transitions: B{f} for each factor
%--------------------------------------------------------------------------
for f = 1:Nf
    B{f} = eye(Ns(f));
end
 
% specify syntax; i.e., the sequence of syntactic (sentence) states
%--------------------------------------------------------------------------
B{3}  = spm_zeros(B{3});
for s = 1:numel(sentence)
    for k = 1:numel(sentence{s})
        for t = 2:numel(sentence{s}{k})
            i = find(ismember(state,sentence{s}{k}(t - 1)));
            j = find(ismember(state,sentence{s}{k}(t)));
            B{3}(j,i) = 1;
        end
        B{3}(j,j) = 1;          % end of phrase: ' '
    end
end
B{3}(11,11) = 1;

% MDP Structure
%--------------------------------------------------------------------------
mdp.T =6;                      % number of updates

mdp.A = a;                      % observation model
mdp.a = a;                      % internal model
mdp.B = B;                      % transition probabilities
mdp.b = B;                      % transition probabilities
mdp.D = D;                      % prior over initial states
mdp.o = [];

mdp.VOX   = [0,0,0];
mdp.label = label;
mdp.tau   = 4;                  % time constant of belief updating
mdp.erp   = 1;                  % initialization
mdp.chi = 100;  
MDP       = spm_MDP_check(mdp);

clear A B D mdp label

%% second level (Conceptual Level)
%==========================================================================
label.factor{1}  = 'spoken word';  label.name{1}  ={'square', 'green', 'blue', 'red', 'triangle'}; 
label.factor{2}  = 'target word';  label.name{2}  ={'square', 'green', 'blue', 'red', 'triangle'}; 
label.factor{3}  = 'context';    label.name{3}  = {'question', 'answer', 'feedback'};

% prior beliefs about initial states D - forming the basis of syntax
%--------------------------------------------------------------------------
for i = 1:numel(label.factor)
    n    = numel(label.name{i});
    D{i} = ones(n,1)/n;
end

% known initial states
%--------------------------------------------------------------------------
D{3}(1)  = 128;

% probabilistic mapping from hidden states to outcomes: A
%--------------------------------------------------------------------------
Nf    = numel(D);
Ng = numel(MDP.D);
for f = 1:Nf
    Ns(f) = numel(D{f}); 
end

label.modality{1} = 'audition';   label.outcome{1}  = [MDP.label.name{1}];
label.modality{2} = 'feedback.';   label.outcome{2}  = MDP.label.name{2};
label.modality{3} = 'syntax'; label.outcome{3}  = MDP.label.name{3}(find(MDP.D{3}));
                                                                                    % finding the appropriate places where the sentence can start from

% Likelihood mappings:
for f = 1:Ng
    No(f) = numel(label.outcome{f});   
end

for g = 1:Ng
    A{g} = zeros([No(g),Ns]); 
end

e = 1e-10;
n4 = length(familiar);

for f1 = 1:Ns(1) % spoken word {'square', 'green', 'blue', 'red', 'triangle'}; 
    for f2 = 1:Ns(2) % target word {'square', 'green', 'blue', 'red', 'triangle'}; 
        for f3 = 1:Ns(3) % narrative {'go','question','answer', feedback}
            
                                    % indices
                                    %--------------------------------------
                                    j = {f1,f2,f3};
                                    
                                    % Audition: 
                                    %---------------------------------------
                                     if (f3 == 1)
                                        A{1}(1:n4,f1, :,f3) = ones(n4,n4)*e+eye(n4,n4);

                                    elseif (f3 == 2 || f3 == 3 ) 
                                            A{1}(1:n4,:,f2,f3) =ones(n4,n4)*e+eye(n4,n4);
  
                                     end     
                                          
                                     % Feedback:
                                     %---------------------------------------
                                     if f3 ~= 3
                                         A{2}(3,:,:,f3) = 1;
                                     else
                                         if f2 == f1
                                             A{2}(1,j{:}) = 1;
                                         elseif (f2 ~= f1)
                                                A{2}(2,j{:}) = 1;    
                                         end
                                     end
                                     
                                     % Syntax:   {Q1, reply,feedback,??};
                                     % narrative {'question','answer', feedback}
                                    %---------------------------------------
                                    if f3 == 3
                                        A{3}(3,j{:}) = 1;
                                        
                                    elseif (f3 == 1)
                                        A{3}(1,j{:}) =1;
                                        
                                    elseif f3 == 2
                                        A{3}(4,j{:}) = .05;  
                                        A{3}(2,j{:}) = .95;  
                                        
                                    end
        end
    end
end

for g = 1:Ng
    a2{g} = 10*A{g};
end


% controlled transitions: B{f} for each factor
%--------------------------------------------------------------------------
for f = 1:Nf
    B{f} = eye(Ns(f));
    label.action{f} = {'stay'};
end
 
% Control state B(3): {'ready','question','answer', feedback}
%--------------------------------------------------------------------------
% Question to answer, Question to Answer  OR Question to Question; Answer
% to Answer
B{3}(1,:) = 0;
B{3}(2:3,1:2,1) = eye(2);

%B{3}(:,:,2) = eye(4);
%B{3}(:,1,2) = circshift(B{3}(:,1,2),1); 

 
% Control state B(2)
%--------------------------------------------------------------------------
B{1} = zeros(n4);
for i = 1:n4
    B{1}(i,:,i) = 1;
    label.action{1} = [label.name{1}];
end 
    
for f = 1:Nf
    b{f} = 10*B{f};
end

% 

% allowable policies (time x policy x factor): ready, question, answer,
%                                                                        feedback)
%--------------------------------------------------------------------------
V         = ones(2,n4,Nf);
V(:,:,1) =     [1 2 3 4 5; 
                    1 2 3 4 5;
                ];
V(:,:,2) = 1;
V(:,:,3) = 1;

% priors: (utility) C: A{3}: {Q1,Q2, go, reply,feedback,??};
%--------------------------------------------------------------------------
c = 1/16;

for g = 1:Ng
    C{g}  = zeros(No(g),3);
end
C{2}(1,:) =  5;              % does want to be right
C{2}(2,:) = -10;             % does not want to be wrong

C{3}(1,1) = c;        % does not want to speak before time point 3
C{3}(4,3) = -c;            % does not want to be unsure when replying
      


% MDP Structure
%--------------------------------------------------------------------------
mdp.FCN   = @spm_questions_plot;

mdp.label = label;              % names of factors and outcomes
mdp.tau   = 2;                  % time constant of belief updating
mdp.erp   = 1;                  % initialization

mdp.T = 3;                      % question, answer, feedback
mdp.V = V;                      % allowable policies

% : likelihood probabilities
mdp.A = A;                      % observation model
mdp.a = a2;                      % internal model

% : transition probabilities
mdp.B = B;                      % observation model
mdp.b = b;                      % internal model

mdp.C = C;                      % preferred outcomes
mdp.D = D;                      % prior over initial states (context)
mdp.o = [];                     % outcomes

mdp       = spm_MDP_check(mdp);

mdp.MDP   = MDP;
mdp.link = spm_MDP_link(mdp);   % map outputs to initial (lower) states
mdp       = spm_MDP_check(mdp);

% factor graph
%==========================================================================
spm_MDP_factor_graph(mdp);

return mdp 

