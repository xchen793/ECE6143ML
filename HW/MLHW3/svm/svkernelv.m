function k = svkernelv(ker,u,v)
%SVKERNELV kernel for Support Vector Methods
%
%  Usage: k = svkernelv(ker,u,v)
%
%  Parameters: ker - kernel type
%              u,v - kernel arguments
%              k   - vector of tensor product components
%
%  Values for ker: 'linear'  -
%                  'poly'    - p1 is degree of polynomial
%                  'rbf'     - p1 is width of rbfs (sigma)
%                  'sigmoid' - p1 is scale, p2 is offset
%                  'spline'  -
%                  'bspline' - p1 is degree of bspline
%                  'fourier' - p1 is degree
%                  'erfb'    - p1 is width of rbfs (sigma)
%                  'anova'   - p1 is max order of terms
%              
%  Author: Steve Gunn (srg@ecs.soton.ac.uk)

  if (nargin < 1) % check correct number of arguments
     help svkernel
  else
     
    global p1 p2;

    % could check for correct number of args in here
    % but will slow things down further
    switch lower(ker)
      case 'curvspline'
        k = u.*v.*min(u,v) - ((u+v)/2).*(min(u,v)).^2 + (1/3)*(min(u,v)).^3;
      case 'anovaspline1'
        k = u.*v + u.*v.*min(u,v) - ((u+v)/2).*(min(u,v)).^2 + (1/3)*(min(u,v)).^3;
      case 'anova'
        k = u.*v + (1/2)*u.*v.*min(u,v) - (1/6)*(min(u,v)).^3;
      case 'anovaspline2'
        k = u.*v + (u.*v).^2 + (u.*v).^2.*min(u,v) - u.*v.*(u+v).*(min(u,v)).^2 + (1/3)*(u.^2 + 4*u.*v + v.^2).*(min(u,v)).^3 - (1/2)*(u+v).*(min(u,v)).^4 + (1/5)*(min(u,v)).^5;
      case 'anovaspline3'
        k = u.*v + (u.*v).^2 + (u.*v).^3 + (u.*v).^3.*min(u,v) - (3/2)*(u.*v).^2.*(u+v).*(min(u,v)).^2 + u.*v.*(u.^2 + 3*u.*v + v.^2).*(min(u,v)).^3 - (1/4)*(u.^3 + 9*u.^2.*v + 9*u.*v.^2 + v.^3).*(min(u,v)).^4 + (3/5)*(u.^2 + 3*u.*v + v.^2).*(min(u,v)).^5 - (1/2)*(u+v).*(min(u,v)).^6 + (1/7)*(min(u,v)).^7;
      case 'anovabspline'
        k = 0;
        for r = 0: 2*(p1+1)
          k = k + (-1)^r*binomial(2*(p1+1),r)*(max(0,u-v + p1+1 - r)).^(2*p1 + 1);
        end
      otherwise
        k = u.*v;
    end

  end
