function amazon_run_exp()
global mod_name;
mod_name = '';

SOURCE_KNOWLEDGE = 'RBM'; % Representation from source knowledge
TARGET_KNOWLEDGE = 'RBM'; % Adaptation base in target domain

% This will get the knowledge from source domain, saved in mod_name
if strcmp(SOURCE_KNOWLEDGE,'RBM')
    amazon_rbm();
elseif strcmp(TARGET_KNOWLEDGE,'dAE')
    amazon_ae();
elseif strcmp(TARGET_KNOWLEDGE,'SC')
    amazon_sc();
elseif strcmp(TARGET_KNOWLEDGE,'NMF')
    amazon_nmf();
end

% This will use mod_name to get the knowlege for adaptation in new domain
if strcmp(TARGET_KNOWLEDGE,'RBM')
    amazon_ast_rbm();
elseif strcmp(TARGET_KNOWLEDGE,'dAE')
    amazon_ast_ae();
elseif strcmp(TARGET_KNOWLEDGE,'SC')
    amazon_ast_sc();
elseif strcmp(TARGET_KNOWLEDGE,'NMF')
    amazon_ast_nmf();
end

end

