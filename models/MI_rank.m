function [mi,inx] = MI_rank(model,data,label,order,dat_row_order)
% Rank feature detectors using mutual information
 
if ischar(data)
        data  = get_data_from_file(data);        
        if nargin>4 && dat_row_order, data = data'; end
end

hidI = bsxfun(@plus,model.W'*data,model.hidB);
if ~isempty(label)
        hidI = hidI + model.U(:,label);
end
hidP = logistic(hidI);
MI = mean(bsxfun(@times,hidP,log2(bsxfun(@rdivide,hidP,mean(hidP,2)))));
hidP = 1-hidP;
MI = MI +  mean(bsxfun(@times,hidP,log2(bsxfun(@rdivide,hidP,mean(hidP,2)))));
[mi,inx] = sort(MI,order);
end
