function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

values = [0.01 0.03 0.1 0.3 1 3 10 30];

best_error = -1;  
C = -1;
sigma = -1;

for curr_C = values
    for curr_sigma = values
        model = svmTrain(X, y, curr_C, @(x1, x2) gaussianKernel(x1, x2, curr_sigma));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        
        if best_error == -1 || error < best_error
            best_error = error
            C = curr_C
            sigma = curr_sigma
        endif
    endfor
endfor

end
