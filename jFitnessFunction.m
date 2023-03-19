% Notation: This fitness function is for demonstration 

function [cost, ACC, Results] = jFitnessFunction(X_train, X_test, y_train, y_test , X)
if sum(X == 1) == 0
   
  cost = 1;
  ACC=0;
  Results.Sensitivity=0;
    Results.Specificity=1;
    Results.Precision=2;
    Results.FalsePositiveRate=3;
    Results.F1_score=4;
    Results.MatthewsCorrelationCoefficient=5;
    Results.Kappa=6;



else
    
  [cost, ACC, Results] = jwrapperKNN( X_train (:,X==1), X_test (:,X==1), y_train, y_test , X );
end
end


function [error, ACC, Results] = jwrapperKNN(X_train, X_test, y_train, y_test , X )
%---// Parameter setting for k-value of KNN //
 

xtrain = table2array(X_train);
ytrain = table2array(y_train ); 
xvalid = table2array(X_test); 
yvalid = table2array(y_test); 
k = 1;
Model     = fitcknn(xtrain,ytrain,'NumNeighbors',k);  
pred      = predict(Model,xvalid);
num_valid = length(yvalid); 
correct   = 0;
for i = 1:num_valid
  if isequal(yvalid(i),pred(i))
    correct = correct + 1;
  end
end
Acc   = correct / num_valid; 
error = 1 - Acc;
%ACC=0;
%X
%P1=sum(X == 1)
%P2=size(X,2)
%error= 0.99 * error_val + 0.01 * (sum(X == 1)/ size(X));

ACC=Acc;
%size(yvalid) 
%size(pred)
[Results]= confusion.getMatrix(yvalid,pred);
% 
%     Results.Sensitivity=0;
%     Results.Specificity=1;
%     Results.Precision=2;
%     Results.FalsePositiveRate=3;
%     Results.F1_score=4;
%     Results.MatthewsCorrelationCoefficient=5;
%     Results.Kappa=6;






end

