function madbase_rbms()

eval(strcat(mfilename,'_setting'));

fprintf('Start running grid search, totally %d experiments. Hit any key to continue \n',...
    size(hidNums,2)*size(lrs,2)*size(lds,2)*size(ps,2));%*size(cs,2)*size(gs,2));
pause();


end

