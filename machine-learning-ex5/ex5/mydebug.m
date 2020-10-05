X = [1 2 3 4];
y = 5;
theta = [0.1 0.2 0.3 0.4]';
[J g] = linearRegCostFunction(X, y, theta, 7)

X = [[1 1 1]' magic(3)];
y = [7 6 5]';
theta = [0.1 0.2 0.3 0.4]';
[J g] = linearRegCostFunction(X, y, theta, 0)

X = [[1 1 1]' magic(3)];
y = [7 6 5]';
theta = [0.1 0.2 0.3 0.4]';
[J g] = linearRegCostFunction(X, y, theta, 7)