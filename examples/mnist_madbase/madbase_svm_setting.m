css      = [1 5 10 50 500 1000];%[0.001 0.005 0.01 0.05 0.1 0.5 1 5 10 50 100 500 1000];
gs      = [0.001 0.005 0.01 0.05 0.1];%[0.0001 0.0005 0.001 0.005 0.01 0.05  0.1 0.5 1 5 10];
Ns       = [10,100,1000,10000,50000];

Ns = Ns(end);
eval('set_exp_paths');

EXP_DIR = strcat(PRJ_DIR,'MNIST_MADBASE',lm,'SVM',lm);    
TGT_DIR = strcat(DAT_DIR,'MADBase',lm);  

if ~exist(EXP_DIR,'dir'), mkdir(EXP_DIR); end
