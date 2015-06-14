sdrop   = 0.7;  % drop-out probability
hidNum  = 500;  % number of additional hidden unit
lr      = 0.7;  % learning rate
mm      = 0;    % momentum
cst     = 0;    % cost

%% Sparsity type
SPARSITY = 'RELU'; % EMIN, KLMIN

%%  sparsity constraint
lds     = 0;
ps      = 0.0001;

as      = 0.1;  % adaptive rate

%% liblinear parameters
css     = 0.001;
pss      = 0.0001;
ess      = 0.00001;


%% data setting
domains = {'books','dvd','electronics','kitchen'};

domain = domains{4};