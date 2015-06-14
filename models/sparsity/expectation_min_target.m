% script to compute sparsity using Expectaion minimization (Lee's)
% require  - model: W, hidB, visB
%          - conf (model & training settings)
%          - input data: visP
%          - batch size: sNum
% return:  - updated model
%          - spasiry error: spr_e
% Son T - 2014

%hidII = hidP;
%current sparsity
pppp = (conf.p - mean(hidtP,2));
hspr = hspr + mean(pppp.^2);
sigmoid_deriv = hidtP.*(1-hidtP);

if isfield(conf,'sparse_w')
    w_diff = conf.lambda*(repmat(pppp',visNum,1).*(visP*sigmoid_deriv')/sNum);
else
    w_diff = 0;
end
h_diff = conf.lambda*(pppp.*(sum(sigmoid_deriv,2)/sNum));

%        if conf.lambda >0                      
%            pppp = (conf.p - sum(hidtP,1)/sNum);           
%            tmpD = lr*conf.lambda*(pppp.*(sum((hidtP.^2).*exp(-hidI),1)/sNum));
%            tmpD(tmask) = 0;
%            model.hidB = model.hidB + tmpD;
%            spr_e = spr_e + sum((conf.p - mean(hidtP)).^2,2);
%        end