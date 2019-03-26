clear;
close all;

% Parameters
lambda_residual = 10000.0;
lambda = 1.0; 
verbose_admm = 'brief';
max_it = 200;

% Load Dictionary
load('../Filters/4d_filters_lightfield.mat', 'd');
d = reshape(d, size(d,1), size(d,2), [], 49);

% Load video data as variable 'b'
% Format should be [x, y, time, indexes]
% Lightfield data can be found in the appropriate folder
% For best results all data should be similarly normalized but not contrast
% normalized.
load('test_data.mat', 'test_full');

for rgb=1:3

for i=1:length(test_full)
    
b = squeeze(test_full{4}(:,:,rgb,1:5,1:5));
sz = size(b);

% Blocks out certain views to reconstruct
MtM = zeros(size(b));
MtM(:,:,1,:) = ones(size(MtM(:,:,1,:)));
MtM(:,:,5,:) = ones(size(MtM(:,:,5,:)));
MtM(:,:,:,5) = ones(size(MtM(:,:,:,5)));
MtM(:,:,:,1) = ones(size(MtM(:,:,:,1)));
MtM(:,:,3,3) = ones(size(MtM(:,:,3,3)));

% Per-Channel Normalization
vmsi = permute(reshape(double(b), [], size(b,3)*size(b,4)), [2 1]);
veam = mean(vmsi,2);
vstd = std(vmsi, 0, 2);
vmsi = (vmsi-repmat(veam, [1 size(vmsi,2)])) ./ repmat(vstd, [1 size(vmsi,2)]);
nb = reshape(vmsi', size(b));

% Sampling matrix
signal_sparse = nb;
signal_sparse( ~MtM ) = 0;

% Quickly interpolate blocked out views to help with contrast normalization
for ss=[2,3,4]
    signal_sparse(:,:,ss,2:end-1) = (signal_sparse(:,:,ss+1,2:end-1)+signal_sparse(:,:,ss-1,2:end-1))/2;
    signal_sparse(:,:,2:end-1,ss) = (signal_sparse(:,:,2:end-1,ss+1)+signal_sparse(:,:,2:end-1,ss-1))/2;
end
signal_sparse(:,:,3,3) = nb(:,:,3,3);

signal_sparse = reshape(signal_sparse, sz(1), sz(2), []);
MtM = reshape(MtM, sz(1), sz(2), []);

% Filter from local contrast normalization
k = fspecial('gaussian',[13 13],3*1.591); 
smooth_init = imfilter(signal_sparse, k, 'same', 'conv', 'symmetric');

fprintf('Doing sparse coding reconstruction.\n\n')
tic();
[z, sig_rec] = admm_solve_conv_weighted_sampling_lf(signal_sparse, d, MtM, lambda_residual, lambda, max_it, 1e-3, [], verbose_admm, smooth_init); 
tt = toc;
fprintf('Done sparse coding! --> Time %2.2f sec.\n\n', tt)

% Un-normalize
vemm = reshape(permute(repmat(veam, [1 size(sig_rec,2)*size(sig_rec,1)]), [2 1]), size(sig_rec));
vssd = reshape(permute(repmat(vstd, [1 size(sig_rec,2)*size(sig_rec,1)]), [2 1]), size(sig_rec));

% Final result for this image and channel
sig_rec_disp = (sig_rec .* vssd + vemm);

end

end
