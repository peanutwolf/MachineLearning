function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

centroids = [];

for k_idx = 1:K
  c_k = 0;
  x_sum = zeros(1,n);
  for x_idx = 1:size(X, 1)
    if idx(x_idx, :) == k_idx
      c_k = c_k + 1;
      x_sum = x_sum + X(x_idx, :);  
    endif
  endfor
  if c_k != 0
    centroids = [centroids;[x_sum/c_k]];
  endif
endfor




% =============================================================


end

