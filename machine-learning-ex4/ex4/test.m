y = [1;2;3;4;5]
m = size(y, 1)
Y = zeros(m, 10);
for i=1:m
  index = y(i)
  Y(i, index) = 1
endfor
