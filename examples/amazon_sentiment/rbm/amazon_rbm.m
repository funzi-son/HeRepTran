function amazon_rbm()
%% Train RBM on unlabelled data

conf.hidNum    = 5000;  % Number of hidden units
conf.eNum      = 100;   % Number of epoch
conf.bNum      = 0;    % Batch number, 0 means it will be decided by the number of training samples
conf.sNum      = 100;  % Number of samples in one batch
conf.gNum      = 1;    % Number of Gibbs sampling
conf.params(1) = lr;  % Learning rate (starting)
conf.params(2) = conf.params(1); % This is unused
conf.params(3) = mm; % Momentum
conf.params(4) = cst; % Weight decay

dat_file = 'unlabelled_',domain,'_data_2kfts.mat';
conf.row_dat = 1; % one data point is one row   

if strcmp(SPARSITY,'RELU'), conf.h_unit  = 'relu'; lds = [0]; ps=[0]; end
for ld = lds
for p = ps
if ld==0 && p>min(ps), continue; end

conf.sparsity  = SPARSITY;% EMIN,KLMIN
conf.cumsparse = 1; % Only for KLMIN, using the expectation of previous batches or not (see the code)
conf.sparse_w  = 1; % Only for EMIN, apply sparsity to w or not(for Lee's approach)
conf.lambda    = ld;    % Sparsity penalty
conf.p         = p;% Sparsity constraint


conf.vis       = 1;
%% Training RBMs
conf.trn_dat_file = dat_file;

 EXP_DIR_ = strcat(EXP_DIR,SPARSITY,lm);
if ~exist(EXP_DIR_,'dir'), mkdir(EXP_DIR_); end
fname = strcat(EXP_DIR_,'rbm_h'...
    ,num2str(conf.hidNum),'_lr',num2str(conf.params(1)),'_mm',num2str(conf.params(3)),...
    '_cst',num2str(cst),'_ld',num2str(ld),'_p',num2str(p),'_trial',num2str(trial),'.mat');

if exist(fname,'file'), continue; end

tic

model = gen_rbm_train(conf);
if GENERATE,
    save(fname,'model');
else
    a = 1;
    save(fname,'a');
end
toc
h_unit = 'binary';
if isfield(conf,'h_unit'), h_unit = conf.h_unit; end
units
if STAT
        %%%%
         %1. Classification using SVM using linear kernel as in Bengio%
        log_file = strrep(fname,'.mat',strcat(domain,'_STL.mat'));
        
        if exist(log_file,'file'), continue; end
        
        trn_dat_file = strcat(TGT_DIR,domain,'_trn_all_dat.mat');
        trn_lab_file = strcat(TGT_DIR,domain,'_trn_all_lab.mat');

        %tst_dat_file = strcat(TGT_DIR,domain,'_tst_dat.mat');
        %tst_lab_file = strcat(TGT_DIR,domain,'_tst_lab.mat');
                %% load data
        vars = whos('-file', trn_dat_file);
        A = load(trn_dat_file,vars(1).name);
        trn_dat = A.(vars(1).name);
        vars = whos('-file', trn_lab_file);
        A = load(trn_lab_file,vars(1).name);
        trn_lab = A.(vars(1).name);


%         vars = whos('-file', tst_dat_file);
%         A = load(tst_dat_file,vars(1).name);
%         tst_dat = A.(vars(1).name);
%         vars = whos('-file', tst_lab_file);
%         A = load(tst_lab_file,vars(1).name);
%         tst_lab = A.(vars(1).name);
             
        clear A;
        
        trn_acc =0;vld_acc=0;vld_acc_=0;vld_av_prec=0;vld_av_recall=0;vld_av_f1=0;tst_acc=0;tst_acc_=0;tst_av_prec=0;tst_av_recall=0;tst_av_f1=0;
           
        count = 0;
        trn_fts = vis2hid(bsxfun(@plus,trn_dat*model.W,model.hidB'));             
%        tst_fts = vis2hid(bsxfun(@plus,tst_dat*model.W,model.hidB'));
             
        clear trn_dat tst_dat;
        %cx = css(1);px = ps(1);ex=es(1);
        for c_ =css
        for p_ = pss
        for e_ = es
            count = count+1;
            fprintf('Grid cell %d ',count);
            %1. Classification using SVM                
            acc = train(trn_lab, sparse(trn_fts),[' -q -v 10 -c ' num2str(c_) ' -p ' num2str(p_) ' -e ' num2str(e_)]);
            if acc>trn_acc, trn_acc = acc; cx = c_;px = p_;ex = e_; end
        end
        end
        end

        slinearmod = train(trn_lab, sparse(trn_fts),[' -q -c ' num2str(cx) ' -p ' num2str(px) ' -e ' num2str(ex)]);
 
        [~, acc, ~] = predict(tst_lab, sparse(tst_fts), slinearmod, ' -q ');
        tst_acc = acc(1);
        disp([trn_acc,tst_acc]);

        logging(log_file,[cx px ex trn_acc]);  
       clear trn_fts tst_fts model;

end

