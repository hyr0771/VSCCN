
clc; clear;
load 3600_xgen_data.mat;
% SET PARAMTER
opts.alpha1 = 5;
opts.alpha2 = 5;
opts.lambda1 = 1;
opts.lambda2 = 0.1;

%% TRAIN AND TEST
[nrow, ~] = size(X);
[test, train] = crossvalind('HoldOut', nrow, 0.7);

X_0 = X(train,:);
Y_0 = Y(train,:);
X_0 = getNormalization(X_0);
Y_0 = getNormalization(Y_0);

XX1 = corr(X');
XX2 = corr(X);
YY1 = corr(Y_0');
YY2 = corr(Y_0);

X_t = X(test,:);
Y_t = Y(test,:);
X_t = getNormalization(X_t);
Y_t = getNormalization(Y_t); 

tic;
[u1, v1, obj] = SCCA_FGL(X_0, Y_0, opts);
tt = toc;
corr_XY = corr(X_t*u1,Y_t*v1);
corr_XY_train = corr(X_0*u1,Y_0*v1);

value1 = X_t*u1;
value2 = Y_t*v1;

%%


