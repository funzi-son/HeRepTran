function madbase_svm()
% Experiment on madbase with SVM

eval(strcat(mfilename,'_setting'));
  

for N = Ns
%% Load data
trn_dat_file = strcat(TGT_DIR,'madbase_trn_dat_file_28x28_',num2str(N),'.mat');
trn_lab_file = strcat(TGT_DIR,'madbase_trn_lab_file_28x28_',num2str(N),'.mat');

vld_dat_file = strcat(TGT_DIR,'madbase_vld_dat_file_28x28_5000_new.mat');
vld_lab_file = strcat(TGT_DIR,'madbase_vld_lab_file_28x28_5000_new.mat');

tst_dat_file = strcat(TGT_DIR,'madbase_tst_dat_file_28x28_10000_new.mat');
tst_lab_file = strcat(TGT_DIR,'madbase_tst_lab_file_28x28_10000_new.mat');
%% load data
vars = whos('-file', trn_dat_file);
A = load(trn_dat_file,vars(1).name);
trn_dat = A.(vars(1).name);
vars = whos('-file', trn_lab_file);
A = load(trn_lab_file,vars(1).name);
trn_lab = A.(vars(1).name);

vars = whos('-file', vld_dat_file);
A = load(vld_dat_file,vars(1).name);
vld_dat = A.(vars(1).name);
vars = whos('-file', vld_lab_file);
A = load(vld_lab_file,vars(1).name);
vld_lab = A.(vars(1).name);

vars = whos('-file', tst_dat_file);
A = load(tst_dat_file,vars(1).name);
tst_dat = A.(vars(1).name);
vars = whos('-file', tst_lab_file);
A = load(tst_lab_file,vars(1).name);
tst_lab = A.(vars(1).name);

clear A;
            % Training SVM
   log_file = strcat(EXP_DIR,'madbase_',num2str(N),'.mat');
    for i5=css
    for i6 = gs
                %1. Classification using SVM                
                 svmmod = svmtrain(trn_lab, trn_dat,[' -q -c ' num2str(i5) ' -g ' num2str(i6)]);
                %[~, accuracy,~] = svmpredict(trn_lab, trn_dat, svmmod);
                trn_acc = 0;%accuracy(1);
                %[predict_label, accuracy, dec_values]
                [output, accuracy,~] = svmpredict(vld_lab, vld_dat, svmmod);
                [~,vld_acc_,vld_av_prec,vld_av_recall,vld_av_f1] = performance_measure(output,vld_lab,'macro');
                vld_acc = accuracy(1);
                %disp([vld_acc_,av_prec,av_recall,av_f1]);
                %pause
                [output, accuracy,~] = svmpredict(tst_lab, tst_dat, svmmod);
                [~,tst_acc_,tst_av_prec,tst_av_recall,tst_av_f1] = performance_measure(output,tst_lab,'macro');
                tst_acc = accuracy(1);                

                logging(log_file,[i5 i6 trn_acc vld_acc vld_acc_ vld_av_prec vld_av_recall vld_av_f1 ...
                    tst_acc tst_acc_ tst_av_prec tst_av_recall tst_av_f1]);
    end % end i6
    end % end i5
end
end

