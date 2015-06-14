function [model,kb] = ast_rbm(conf,kb)
data = get_data_from_file(conf.trn_dat_file);
if isfield(conf,'row_dat') && conf.row_dat, data = data'; end

[visNum,SZ]      = size(data);
hidNum  = conf.hidNum;
kbhNum = size(kb.W,2);

lr    = conf.params(1);

init_vl = (1/max(visNum,hidNum));

model.W     = init_vl*(2*rand(visNum,hidNum)-1);
model.visB  = zeros(visNum,1);   
model.hidB  = zeros(hidNum,1);
kb.hidB     = zeros(kbhNum,1);
kb.A        = ones(kbhNum,1);

DW    = zeros(size(model.W));
DA    = zeros(kbhNum,1);

DHtB   = zeros(hidNum,1);
DHsB   = zeros(kbhNum,1);
DVB   = zeros(visNum,1);

%% Define units for each layer
v_unit = 'binary';
h_unit = 'binary';
if isfield(conf,'v_unit'), v_unit = conf.v_unit; end
if isfield(conf,'h_unit'), h_unit = conf.h_unit; end

fprintf('Start training an RBM: %d %s x %d %s\n',visNum,v_unit,hidNum,h_unit);
units


if hidNum>0
    bNum = conf.bNum;
    if bNum ==0
        if conf.sNum==0
            bNum = 1;
            conf.sNum = SZ; 
        else
            bNum = ceil(SZ/conf.sNum);
        end       
    end
    
    sW = bsxfun(@times,kb.W,kb.A');
%% PCD
if isfield(conf,'pcd')
    if (isfield(conf,'s_dropout') || isfield(conf,'t_dropout'))
        fprintf('This version does not support PCD and Drop-out together');
    else
        visP = rand(visNum,conf.pNum);
        hidtN = vis2hid(bsxfund(@plus,model.W'*visP,model.hidB));
        hidsN = vis2hid(bsxfun(@plus,sW'*visP,kb.hidB));       
        hidtNs =  hid_sample(hidtP);       
        hidsNs =  hid_sample(hidsP);
    end
end

for e=1:conf.eNum
    if e== conf.N+1
        lr = conf.params(2);
    end
    res_e = 0;
    spr_e = 0;
    for b=1:bNum
       sW = bsxfun(@times,kb.W,kb.A');
       % Drop-out
       smask = [];
       tmask = [];
       if isfield(conf,'t_dropout')
        tmask = randperm(hidNum);tmask = tmask(floor(hidNum*conf.t_dropout)+1:end);
       end
       if isfield(conf,'s_dropout')
        smask = randperm(kbhNum);smask = smask(floor(kbhNum*conf.s_dropout)+1:end);       
       end    
       
       inds = (b-1)*conf.sNum+1:min(b*conf.sNum,SZ);
       visP = data(:,inds);       
       sNum = size(visP,2);
       
       %up
       hidI  = bsxfun(@plus,model.W'*visP,model.hidB);
       hidtP = vis2hid(hidI);       
       hidsP = vis2hid(bsxfun(@plus,sW'*visP,kb.hidB));
       
       hidtP(tmask,:) = 0;
       hidsP(smask,:) = 0;
        
       hidtPs =  hid_sample(hidtP);
       hidsPs =  hid_sample(hidsP);
        
       hidtPs(tmask,:) = 0;
       hidsPs(smask,:) = 0;
       
       if ~isfield(conf,'pcd')
        hidtNs = hidtPs;
        hidsNs = hidsPs;
       else
        hidtNs(tmask,:) = 0;
        hidsNs(smask,:) = 0;
       end
       
       for k=1:conf.gNum
           % down
           visN  = hid2vis(bsxfun(@plus,model.W*hidtNs + sW*hidsNs,model.visB));        
           visNs = vis_sample(visN);
           
           hidtN  = vis2hid(bsxfun(@plus,model.W'*visNs,model.hidB));
           hidtN(tmask,:) = 0;
           hidtNs = hid_sample(hidtN);
           hidtNs(tmask,:) = 0;
       
           hidsN =  vis2hid(bsxfun(@plus,sW'*visNs,kb.hidB));
           hidsN(smask,:) = 0;
           hidsNs = hid_sample(hidsN);
           hidsNs(smask,:) = 0;
       end
       
       % Compute MSE for reconstruction              
       res_e = res_e + sum(sqrt(sum((visP - visNs).^2)/visNum))/sNum;
       % Update model
       diff = (visP*hidtP' - visNs*hidtN')/sNum;
       tmpD  = lr*(diff - conf.params(4)*model.W) +  conf.params(3)*DW;
       tmpD(:,tmask) = DW(:,tmask); % Keep the dropout part as before
       DW = tmpD; tmpD(:,tmask) = 0;
       model.W   = model.W + tmpD;
       
       DVB  = lr*sum(visP - visN,2)/sNum + conf.params(3)*DVB;
       model.visB = model.visB + DVB;
       
       tmpD  = lr*sum(hidtP - hidtN,2)/sNum + conf.params(3)*DHtB;
       tmpD(tmask) = DHtB(tmask);DHtB = tmpD;tmpD(tmask) = 0;       
       model.hidB = model.hidB + tmpD;
       
       % Update KB       
       tmpD  = lr*sum(hidsP - hidsN,2)/sNum + conf.params(3)*DHsB;
       tmpD(smask) = DHsB(smask);DHsB = tmpD;tmpD(smask) = 0;
       kb.hidB = kb.hidB + tmpD;
       if conf.a_rate ~=0
           diff = (kb.W'*visP).*hidsP - (kb.W'*visN).*hidsN;
           DA = lr*sum(diff,2)/sNum;
           DA(smask) = 0;
           kb.A = kb.A + conf.a_rate*DA;
       end
       
        if ~strcmp(h_unit,'relu') && isfield(conf,'sparsity') && conf.lambda>0
            if strcmp(conf.sparsity,'EMIN')
                expectation_min_target;
            elseif strcmp(conf.sparsity,'KLMIN')                
                kl_min_target;
            else
                fprintf('No sparsity constraint is set\n');
                continue;
            end            
           model.W = model.W + lr*w_diff;           
           model.hidB = model.hidB + lr*h_diff;
        end
%        if conf.lambda >0                      
%            pppp = (conf.p - sum(hidtP,1)/sNum);           
%            tmpD = lr*conf.lambda*(pppp.*(sum((hidtP.^2).*exp(-hidI),1)/sNum));
%            tmpD(tmask) = 0;
%            model.hidB = model.hidB + tmpD;
%            spr_e = spr_e + sum((conf.p - mean(hidtP)).^2,2);
%        end
    end        
    fprintf('[Epoch %d] res_e = %.5f || spr_e = %.3f \n',e,res_e/bNum,spr_e/bNum);
    if isnan(spr_e/bNum), return; end
end
end
kb.W = bsxfun(@times,kb.W,kb.A');
end

