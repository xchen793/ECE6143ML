clc;
load('teapots.mat');
data = importdata('teapots.mat');
M = mean(data); %calculate the mean of each column
x = data - M; %subtract the mean from the original data
C = cov(x); %calculate the covariance matrix od decentered data x
[V,D] = eig(C); %matrix V whose columns are eigenvectors and diagonal matrix D of eigenvalues
[d, ind] = sort(diag(D), 'descend'); %sort eigenvalue according to its significance
d = d(1:3,:); 
vtop = V(:,ind(1:3));%show top 3 Eigenvectors
X = M + (x*vtop)*vtop';

for i = 1:10
    figure(i);
    colormap gray;
    subplot(1,2,1);
    imagesc(reshape(data(i,:),38,50));
    title('Before PCA');
    axis image;
    subplot(1,2,2);
    imagesc(reshape(X(i,:),38,50));
    title('After PCA');
    axis image;
   
end

 p = sprintf("The least square error is %d.", norm(X-data));
 disp(p)

    




