% Reference MATLAB code for testsvm.html
% Required CVX optimization toolbox! http://cvxr.com/cvx/

X =[-0.4326    1.1909
    3.0000    4.0000
    0.1253   -0.0376
    0.2877    0.3273
   -1.1465    0.1746
    1.8133    2.1139
    2.7258    3.0668
    1.4117    2.0593
    4.1832    1.9044
    1.8636    1.1677];
    
y =[ 1
     1
     1
     1
     1
    -1
    -1
    -1
    -1
    -1]

X(2, 1)= 3; 
X(2, 2)= 4; % makes data nonseparable

% use CVX to train
C = 3;
m = size(X,1);
n = size(X,2);
cvx_begin
    variables w(n) b xi(m)
    minimize 1/2*sum(w.*w) + C*sum(xi)
    y.*(X*w + b) >= 1 - xi;
    xi >= 0;
cvx_end

% print out results
w
b
[y (X*w + b)]

% prints:
% w =
%    -1.1096
%    -0.0590
% b =
%     1.1368
% ans =
%     1.0000    1.5466
%     1.0000   -2.4281
%     1.0000    1.0000
%     1.0000    0.7983
%     1.0000    2.3987
%    -1.0000   -1.0000
%    -1.0000   -2.0688
%    -1.0000   -0.5511
%    -1.0000   -3.6174
%    -1.0000   -1.0000
