function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

m = size(X, 1);
K = size(centroids, 1);
idx = zeros(m, 1);

for i = 1:size(X, 1)
    min_idx = -1;
    min_dist = 0;

    for j = 1:K
        delta = X(i, :) - centroids(j, :);
        curr_dist = sqrt(delta * delta');
        
        if min_idx == -1 || curr_dist < min_dist
            min_idx = j;
            min_dist = curr_dist;
        endif
    endfor
    
    idx(i) = min_idx;
endfor

end

