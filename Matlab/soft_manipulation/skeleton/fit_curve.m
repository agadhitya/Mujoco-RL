function [f, coeffs] = fit_curve(skeletonPts)

curvePtsX = skeletonPts(:,1);
curvePtsY = skeletonPts(:,2);

% Fitting polynomial to skeleton
f=fit(curvePtsX,curvePtsY,'poly3');
    
% curve co-efficients
coeffs = [f.p1, f.p2, f.p3, f.p4];
end