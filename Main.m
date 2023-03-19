%--------------------------------------------------------------------%
%  Equilibrium Optimizer (EO) source codes demo version              %
%--------------------------------------------------------------------%


%---Inputs-----------------------------------------------------------
% feat     : feature vector ( Instances x Features )
% label    : label vector ( Instances x 1 )
% N        : Number of particles
% max_Iter : Maximum number of iterations
% a1       : Parameter 
% a2       : Parameter 
% GP       : Generation rate control parameter 

%---Output-----------------------------------------------------------
% sFeat    : Selected features (instances x features)
% Sf       : Selected feature index
% Nf       : Number of selected features
% curve    : Convergence curve
%--------------------------------------------------------------------


%% Equilibrium Optimizer
clc, clear, close; 
% Benchmark data set 
%load ionosphere.mat; 
%Data={'TicTacToe.mat'};
% Load Data

% X_train = readtable('C:\Users\alomariosa\OneDrive - University of Sharjah\UOS Python projects\Feature selection_Sanad_Speech\Datasets\Speech\FinalData\ravdess\X_train.csv');
% X_test = readtable('C:\Users\alomariosa\OneDrive - University of Sharjah\UOS Python projects\Feature selection_Sanad_Speech\Datasets\Speech\FinalData\ravdess\X_test.csv');
% y_train = readtable('C:\Users\alomariosa\OneDrive - University of Sharjah\UOS Python projects\Feature selection_Sanad_Speech\Datasets\Speech\FinalData\ravdess\y_train.csv');
% y_test = readtable('C:\Users\alomariosa\OneDrive - University of Sharjah\UOS Python projects\Feature selection_Sanad_Speech\Datasets\Speech\FinalData\ravdess\y_test.csv');


X_train = readtable('C:\Users\Scand\OneDrive - University of Sharjah\UOS Python projects\Feature selection_Sanad_Speech\Datasets\Speech\FinalData\EM\X_train.csv')
X_test = readtable('C:\Users\Scand\OneDrive - University of Sharjah\UOS Python projects\Feature selection_Sanad_Speech\Datasets\Speech\FinalData\EM\X_test.csv');
y_train = readtable('C:\Users\Scand\OneDrive - University of Sharjah\UOS Python projects\Feature selection_Sanad_Speech\Datasets\Speech\FinalData\EM\y_train.csv');
y_test = readtable('C:\Users\Scand\OneDrive - University of Sharjah\UOS Python projects\Feature selection_Sanad_Speech\Datasets\Speech\FinalData\EM\y_test.csv');


%Dataset = zeros(30);

% for i=1:1
%    
% Dataset(i) = load(Data{1});
% end


% Parameter setting
N        = 40; 
max_Iter = 100;
run = 10;

%for RUN = 1:run


% for i=1:size(Dataset)
% X = Dataset(i);
% label = X.Tictactoe(:, end);
% feat = X.Tictactoe(:, 1:(end-1));
% % Set 20% data as validation set
% ho = 0.2; 
% % Hold-out method
% HO = cvpartition(label,'HoldOut',ho);

% 
% % Equilibrium Optimizer
%[Sf,Nf,curve] = AOA(X_train, X_test, y_train, y_test , N,max_Iter,  run);
%[Sf,Nf,curve] = jMarinePredatorsAlgorithm_Basic(X_train, X_test, y_train, y_test , N,max_Iter,  run);

[SMA] = jSlimeMouldAlgorithm_EM(X_train, X_test, y_train, y_test , N,max_Iter,  run);


% % Plot convergence curve
% plot(1:max_Iter,curve(1:100));
% xlabel('Number of iterations');
% ylabel('Fitness Value');
% title('AOA'); grid on;
% hold on;
% display(' -------------------------------')
% 
% [sFeat,Sf,Nf,curve, WSOFF] = WSO(feat,label,N,max_Iter,HO, WSOFF, RUN);
% 
% % Plot convergence curve
% plot(1:max_Iter,curve(1:100));
% xlabel('Number of iterations');
% ylabel('Fitness Value');
% title('WSO'); grid on;
% hold on;
% display(' -------------------------------')
% 
% %Parameter settings
% [sFeat,Sf,Nf,curve, EOFF] = jEO(feat,label,N,max_Iter,HO, EOFF, RUN);
% 
% plot(1:max_Iter,curve(1:100));
% xlabel('Number of iterations');
% ylabel('Fitness Value');
% title('EO'); grid on;
% hold on;


display(' -------------------------------')

%[sFeat,Sf,Nf,curve] = BAT(feat,label,N,max_Iter,HO,run);


%[sFeat,Sf,Nf,curve, BATFF] = GWO(feat,label,N,max_Iter,HO run);

%[Sf,Nf,curve] = SSA(X_train, X_test, y_train, y_test , N,max_Iter, run);



%plot convergence curve
plot(1:max_Iter,curve(1:100));
xlabel('Number of iterations');
ylabel('Fitness Value');
legend('AOA', 'WSO', 'EO', 'BAT')  
title('BAT'); grid on;




%end

%end


