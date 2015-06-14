function amazon_ast_rbm()
global mod_name;
eval(strcat(mfilename,'_setting'));

% Load representation knowledge from source domain
load(mod_name)
load(strcat(MOD_DIR,mod_name));      	
model_init.W = model.W;
clear model;

conf.hidNum = hidNum;
conf.eNum   = 100;
conf.bNum   = 0;
conf.sNum   = 100;
conf.gNum   = 1;
conf.params = [lr lr mm cst];
conf.lambda = 0;
conf.a_rate = i5;
conf.v_unit    = 'binary';
conf.h_unit    = 'binary';


if strcmp(SPARSITY,'RELU'), conf.h_unit  = 'relu'; lds = [0]; ps=[0]; end
conf.sparsity = SPARSITY;
conf.s_dropout = sdrop;
conf.N      = 10;
conf.row_dat = 1;

dat_file = strcat(TGT_DIR,domain,'_trn_dat.mat');
conf.trn_dat_file = dat_file;

if isfield(conf,'h_unit'), h_unit = conf.h_unit; end
if isfield(conf,'v_unit'), v_unit = conf.v_unit; end

%% Training
tic
units
[model,smodel] = ast_rbm(conf,model_init);
toc

%% Classification
trn_dat_file = strcat(TGT_DIR,domain,'_trn_dat.mat');
trn_lab_file = strcat(TGT_DIR,domain,'_trn_lab.mat');

tst_dat_file = strcat(TGT_DIR,domain,'_tst_dat.mat');
tst_lab_file = strcat(TGT_DIR,domain,'_tst_lab.mat');
%% load data
trn_dat = get_data_from_file(trn_dat_file);
trn_lab = get_data_from_file(trn_lab_file)';
tst_dat = get_data_from_file(tst_dat_file);
tst_lab = get_data_from_file(tst_lab_file)';

trn_acc =0;vld_acc=0;vld_acc_=0;vld_av_prec=0;vld_av_recall=0;vld_av_f1=0;tst_acc=0;tst_acc_=0;tst_av_prec=0;tst_av_recall=0;tst_av_f1=0;
            
trn_fts = vis2hid(bsxfun(@plus,trn_dat*[smodel.W model.W],[smodel.hidB;model.hidB]'));
tst_fts = logistic(bsxfun(@plus,tst_dat*[smodel.W model.W],[smodel.hidB;model.hidB]'));
clear trn_dat tst_dat;   
css     = 0.001;
pss      = 0.0001;
ess      = 0.00001;
slinearmod = train(trn_lab, sparse(trn_fts),[' -q -c ' num2str(css) ' -p ' num2str(pss) ' -e ' num2str(ess)]);
[~, acc, ~] = predict(tst_lab, sparse(tst_fts), slinearmod, ' -q ');
tst_acc = acc(1);
disp(tst_acc);
clear trn_fts tst_fts model;

end

