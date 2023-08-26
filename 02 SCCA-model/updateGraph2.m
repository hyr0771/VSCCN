function [E] = updateGraph2(p, flag)
% --------------------------------------------------------------------
% Update the weight matrix, degree matrix and the laplacian matrix
% --------------------------------------------------------------------
% Input:
%       - beta, coeffients
%       - W, weight matrix
%       - flag, the type of laplacian matrix L, e.g. +1 or -1
% Output:
%       - E, edge matrix E
%------------------------------------------

ncol = p;
iedge = 0;

switch flag
    case 'GGL'% GGL penalty
        E = zeros(p*(p-1),p);
        for i = 1:p
            for j = 1:p
                if i ~= j
                    iedge = iedge+1;
                    E(iedge,i) = 1;
                    E(iedge,j) = 1;
                end
            end
        end
    case 'FGL' % FGL penalty
        E = zeros(2*(p-1),p);
        for i = 1:p-1
            j = i+1;
            E(2*i-1,i) = 1;
            E(2*i-1,j) = 1;
            E(2*i,i) = 1;
            E(2*i,j) = 1;
        end
end