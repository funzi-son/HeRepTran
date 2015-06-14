function amazon_ast_rbm()
global mod_name;
eval(strcat(mfilename,'_setting'));

load(mod_name)
    load(strcat(MOD_DIR,mod_name));      	
    model_init.W = model.W;
    clear model;
for i0 = sdrops
for i1 = hidNums    
for i2 = lrs    
for i3 = mms     
for i4 = csts   
    EXP_DIR_ = strcat(EXP_DIR,SPARSITY,lm);
if ~exist(EXP_DIR_,'dir'), mkdir(EXP_DIR_); end
    lname = strcat(EXP_DIR_,'kb',num2str(i0),'_h'...
    ,num2str(i1),'_lr',num2str(i2),'_mm',num2str(i3),...
    '_cst',num2str(i4),'_',mod_name,'_',domain,'_trial',num2str(trial),'.mat');
for i5 = as

conf.hidNum = i1;
conf.eNum   = 100;
conf.bNum   = 0;
conf.sNum   = 100;
conf.gNum   = 1;
conf.params = [i2 i2 i3 i4];
conf.lambda = 0;
conf.a_rate = i5;
conf.v_unit    = 'binary';
conf.h_unit    = 'binary';

if strcmp(SPARSITY,'RELU'), conf.h_unit  = 'relu'; lds = [0]; ps=[0]; end
conf.sparsity = SPARSITY;
conf.s_dropout = i0;
conf.N      = 10;
conf.row_dat = 1;

dat_file = strcat(TGT_DIR,domain,'_trn_all_2kfts_dat.mat');
conf.trn_dat_file = dat_file;

fname = strrep(lname,'.mat',strcat('_as',num2str(i5),'.mat'));
if exist(fname,'file'), continue; end

tic
h_unit = 'binary';
v_unit = 'binary';
if isfield(conf,'h_unit'), h_unit = conf.h_unit; end
if isfield(conf,'v_unit'), v_unit = conf.v_unit; end
units
[model,smodel] = ast_rbm(conf,model_init);
if GENERATE,
    save(fname,'model');
else
    a = 1;
    save(fname,'a');
end
toc

if STAT
            log_file = strrep(lname,'.mat','_ATSDROP.mat');

            %if exist(log_file,'file'), continue; end

            trn_dat_file = strcat(TGT_DIR,domain,'_trn_all_2kfts_dat.mat');
            trn_lab_file = strcat(TGT_DIR,domain,'_trn_all_2kfts_lab.mat');

            %tst_dat_file = strcat(TGT_DIR,domain,'_tst_dat.mat');
            %tst_lab_file = strcat(TGT_DIR,domain,'_tst_lab.mat');
                    %% load data
            vars = whos('-file', trn_dat_file);
            A = load(trn_dat_file,vars(1).name);
            trn_dat = A.(vars(1).name);
            vars = whos('-file', trn_lab_file);
            A = load(trn_lab_file,vars(1).name);
            trn_lab = A.(vars(1).name);


%             vars = whos('-file', tst_dat_file);
%             A = load(tst_dat_file,vars(1).name);
%             tst_dat = A.(vars(1).name);
%             vars = whos('-file', tst_lab_file);
%             A = load(tst_lab_file,vars(1).name);
%             tst_lab = A.(vars(1).name);

            clear A;

            trn_acc =0;vld_acc=0;vld_acc_=0;vld_av_prec=0;vld_av_recall=0;vld_av_f1=0;tst_acc=0;tst_acc_=0;tst_av_prec=0;tst_av_recall=0;tst_av_f1=0;

            count = 0;
            
            trn_fts = vis2hid(bsxfun(@plus,trn_dat*[smodel.W model.W],[smodel.hidB;model.hidB]'));
            %tst_fts = logistic(bsxfun(@plus,tst_dat*[smodel.W model.W],[smodel.hidB;model.hidB]'));

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
% 
%             slinearmod = train(trn_lab, sparse(trn_fts),[' -q -c ' num2str(cx) ' -p ' num2str(px) ' -e ' num2str(ex)]);
% 
%             [~, acc, ~] = predict(tst_lab, sparse(tst_fts), slinearmod, ' -q ');
%             tst_acc = acc(1);
%             disp([trn_acc,tst_acc]);

            logging(log_file,[i5 cx px ex trn_acc]);  
            clear trn_fts tst_fts model;
 %       end
end

end % as    
end % ps
end % lambdas
end %learningrates
end %hidnums
end % drops
end

end

