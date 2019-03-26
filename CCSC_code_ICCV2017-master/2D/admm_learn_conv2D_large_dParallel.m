function [ d_res, z_res, DZ, iterations ] = admm_learn_conv2D_large(b, kernel_size, ...
                    lambda_residual, lambda_prior, ...
                    max_it, tol, ...
                    verbose, init)
    
    %Kernel size contains kernel_size = [psf_s, psf_s, k]
    psf_s = kernel_size(1);%11
    k = kernel_size(end);%100
    sb = size(b);%100 100 1 6
    n = sb(end);%ȡͼƬ��������6
    ni=2; %ÿ�δ�����ͼƬ       
    N = n/ni; %3
    
    
    %-------------By Paulin-------------%
    %��ʼ��yk
    
    %��������Ӧ������Ҫ�ı���
    rhoF=500;%filter subproblem rho ��ʼ��Ϊ500
    rhoC=50;%coefficient subproblem rho ��ʼ��Ϊ50
    %�ֱ���primal�в��dual�в�
    %filter subproblem��
    Primal_filter=Inf;%(di-yk)/max{di,yk}д��Ҳû�ã����ܲ��ԡ�
    Dual_filter=Inf;%(yk+1-yk)/||nameda||2
    epri_filter=0;
    edua_filter=0;
    %coefficient subproblem��
    Primal_coefficient=Inf;
    Dual_coefficient=Inf;
    epri_coefficient=0;
    edua_coefficient=0;
    %����taoΪrhomlt��sporco�е�һ��
    %Ҫ�õ���һЩ����
    rhomlt_filter=0;
    rhoscaling=100;
    rhorsdltarget=1+(18.3).^(log10(lambda_prior)+1);
    rhorsdlratio=1.2;
    %----------------------------------%
                
    %PSF estimation
    psf_radius = floor( psf_s/2 );%5
    size_x = [sb(1:end - 1) + 2*psf_radius, n];%110 110 6
    size_z = [size_x(1:end - 1), k, n];%110 110 100 6
    size_z_crop = [size_x(1:end - 1), k, ni];%110 110 100 2 ϡ�����Ĵ�С
    size_d_full = [size_x(1:end - 1), k]; %110 110 100
    
    %-----------------by Paulin ��ά�Ⱥ������ֹͣ����--------------------%
    Nx_filter=prod(size_z_crop);
    Nx_coefficient=prod(size_z_crop);
    AbsStopTol=0;
    RelStopTol=1e-6;
    %---------------------------------------------%
     
    lambda = [lambda_residual, lambda_prior];%1.0 1.0

    B = padarray(b, [psf_radius, psf_radius, 0], 0, 'both');%��ͼ����Χ���0 ��110 110
    B_hat = fft2(B);%����Ҷ�仯 ��Сû�иı�  110 110 6
    %��ͼ��ֿ�
    for nn=1:N
        Bh{nn} = B_hat(:,:,(nn-1)*ni + 1:nn*ni) ;%110 110 2
    end 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Proximal Operators��������
    %ProxDataMasked = @(u, B, theta) (B + 1/theta * u ) ./ ( 1 + 1/theta ); 
    ProxSparse = @(u, theta) max( 0, 1 - theta./ abs(u) ) .* u; %����coefficient��prox
    ProxKernelConstraint = @(u) KernelConstraintProj( u, size_d_full, psf_radius);%����filter��prox     
    % Objective
    objective = @(z, d) objectiveFunction( z, d, b, lambda_residual, lambda_prior, psf_radius, size_z, size_x );      
 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %��d���д���
    d = padarray( randn(kernel_size), [size_x(1) - kernel_size(1), size_x(2) - kernel_size(2),0], 0, 'post');%��ÿһά���һ��Ԫ�ص����
    d = circshift(d, -[psf_radius, psf_radius, 0] );%����������λת���������о������ʽ
    
    d_hat = fft2( d );%���˲������о��
    dup = repmat({d_hat},N,1);%����Ԥ�����õĿ������˲���ƽ�̳ɶ�Ӧ����״����Ҷ����
    D = repmat({d}, N,1);%D��ʱ���µ�d
    %��z���д���
    z = randn(size_z);%100��110x110��z��110 110 100 6���еģ�ʹ����̬�ֲ���ʼ��ϡ��ϵ������
    z_hat = fft2( z );%���и���Ҷ�仯
       
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %�������
    if  strcmp( verbose, 'all')
        iterate_fig = figure();
        filter_fig = figure();
        display_func(iterate_fig, filter_fig, d, z_hat, b, size_x, size_z_crop, psf_radius, 0);
    end
    if strcmp( verbose, 'brief') || strcmp(verbose, 'all')
        obj_val = objective(z, d);
        fprintf('Iter %d, Obj %3.3g, Diff %5.5g\n', 0, obj_val, 0)
        obj_val_filter = obj_val;
        obj_val_z = obj_val;
    end
    
    %Save all objective values and timings
    iterations.obj_vals_d = [];
    iterations.obj_vals_z = [];
    iterations.tim_vals = [];
    %%iterations.it_vals = [];
     
    %Save all initial vars
    iterations.obj_vals_d(1) = obj_val_filter;
    iterations.obj_vals_z(1) = obj_val_z;
    iterations.tim_vals(1) = 0;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Iteration for local back and forth
    max_it_d = 10;%d ����10��
    max_it_z = 10;%z ����10�Σ�Ȼ����һ���ܵĵ�������Ϊ20��

    %%%%%%%%%%%%% d specific
    Dbar = zeros(size_d_full);%110 110 100��С��0����
    Udbar = zeros(size_d_full);%    
    %----------------By Paulin-----------------%
    Yprv_filter=zeros(size_d_full);
    Yprv_coefficient=zeros(size_z); 
    
    %-------------------------------------------%
    %D = repmat({ zeros(size_k_full) },N,1) ;
    d_D = repmat({zeros(size_d_full)},N,1);%��Dbarƽ�̰���N ��D��ʼ��Ϊ0
    %%%%%%%%%%%%%%%%%% z specific %%%%%%%

    d_Z =  zeros(size_z);    %��z��ʼ��Ϊ0��110 110 100 6
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %Iterate
    for i = 1:max_it              
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
        %obj_val_min = min(obj_val_filter, obj_val_z);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% D
        tic;
        for nn=1:N
            fprintf('Starting D preprocessing iterations: %d! \n', nn);
            zup{nn} = z_hat(:,:,:,(nn-1)*ni + 1:nn*ni) ;%��z�ֿ�
            [zhat_mat{nn}, zhat_inv_mat{nn}] = precompute_H_hat_D(zup{nn}, size_z_crop, rhoF); %gammas_D(2)/gammas_D(1)
            %�����500��rho�������Ż�D�ı��ʽ�У�����z���Ǹ����ӣ���ZtZ+pI��-1�������õ���rho
        end
        t_kernel = toc;
        fprintf('Starting D iterations after preprocessing! \n')
         
        for i_d = 1:max_it_d
           if Primal_filter>epri_filter&&Dual_filter>edua_filter
            d_old = D{1};
            tic;
            u_D2 = ProxKernelConstraint( Dbar + Udbar );%yk+1=prox(dk+1 + named k)�����yk+1��������£���Ҫ��ʼ��һ��yk
            for nn = 1:N
                %fprintf('Iter D %d, nn %d\n', i_d, nn);
                d_D{nn} = d_D{nn} + (D{nn} - u_D2);
                ud_D{nn} = fft2( u_D2 - d_D{nn} ) ;
                dup{nn} = solve_conv_term_D(zhat_mat{nn}, zhat_inv_mat{nn}, ud_D{nn}, Bh{nn}, rhoF, size_z_crop);
                D{nn} = real(ifft2( dup{nn} ));
            end
            Dbar =0; Udbar = 0;
            for nn=1:N
                Dbar = Dbar + D{nn};
                Udbar = Udbar + d_D{nn};
            end
            Dbar = (1/N)*Dbar;
            Udbar = (1/N)*Udbar;
            
            t_kernel_tmp = toc;
            t_kernel = t_kernel + t_kernel_tmp;
            
            d_diff = D{1} - d_old;
            if strcmp(verbose, 'brief')
                obj_val_filter = objective(z, D{1});
                fprintf('Iter D %d, Obj %3.3g, Diff %5.5g\n', i_d, obj_val_filter, norm(d_diff(:),2)/ norm(D{1}(:),2));
            end
            if (norm(d_diff(:),2)/ norm(D{1}(:),2) < tol)
                break;
            end
            %----------------------By Paulin-----------------------%
            %��filter��primal�в��dual�в�
            %primal=(di-y)/max{di,yk}
            %dual=yk+1-yk/||named k||
%             if i_d >= 2
%             Primal_filter=norm(vec(Udbar))/max(norm(Dbar(:)),norm(u_D2(:)))%�ϴ�
%             %norm(vec(u_D2));%��һ��u_D2��0����һ������0
%             Dual_filter=norm(vec(Yprv_filter-u_D2))/norm(Udbar(:))%��С �����ǵ�һ��Ϊ0��
%             %����ֹͣ������
%             epri_filter=sqrt(Nx_filter)*AbsStopTol/max(norm(Dbar(:)),norm(u_D2(:)))+RelStopTol;
%             edua_filter=sqrt(Nx_filter)*AbsStopTol/norm(Udbar(:))+RelStopTol;
%              Yprv_filter=u_D2;                    
%             %Ȼ�����tao��ֵ
%             rhomlt_filter=sqrt(Primal_filter/(Dual_filter*rhorsdltarget))
%             if rhomlt_filter <1, rhomlt_filter=1/rhomlt_filter;end
%             if rhomlt_filter>rhoscaling,rhomlt_filter=rhoscaling;end
%             rsf_filter=1;
%             if Primal_filter > rhorsdltarget*rhorsdlratio*Dual_filter, rsf_filter=rhomlt_filter;end
%             if Dual_filter > (rhorsdlratio/rhorsdltarget)*Primal_filter,rsf_filter=1/rhomlt_filter;end
%             rhoF=rsf_filter*rhoF;
            %Udbar=Udbar/rsf;
           % end
            %-------------------------------------------------%
        end
        end         
        if  strcmp( verbose, 'all')
            display_func(iterate_fig, filter_fig, D{1}, z_hat, b, size_x, size_z_crop, psf_radius, i);
        end        
        
        %%%%% Z 
        tic;
        %for nn=1:N
            fprintf('Starting Z preprocessing iterations:! \n');
            [dhat_flat, dhatTdhat_flat] = precompute_H_hat_Z(fft2(D{1}), size_x);
            dhatT_flat = repmat(  conj(dhat_flat.'), [1,1,n] ); 
        %end
        t_vars = toc;
        for i_z = 1:max_it_z
            z_old = z;
            tic;
            u_Z2 = ProxSparse( z + d_Z, lambda(2)/50 );%����ĺ���һ����theta���Ȳ��ܸ�rho����û�й�ϵ
            d_Z = d_Z + (z - u_Z2);
            ud_Z = fft2( u_Z2 - d_Z ) ;
            z_hat = solve_conv_term_Z(dhatT_flat, dhatTdhat_flat, ud_Z, B_hat, rhoC, size_z);
            z = real(ifft2( z_hat ));
            t_vars_tmp = toc;
            t_vars = t_vars + t_vars_tmp;
            z_diff = z - z_old;
            %-----------------By Paulin---------------------%
%             Primal_coefficient =norm(vec(z - u_Z2))/max(norm(z(:)),norm(u_Z2(:)));
%             Dual_coefficient =norm(vec(Yprv_coefficient-u_Z2))/norm(d_Z(:));
%             Yprv_coefficient=u_Z2;
%             rhomlt_coefficient=sqrt(Primal_coefficient/(Dual_coefficient*rhorsdltarget));
%             if rhomlt_coefficient<1,rhomlt_coefficient=1/rhomlt_coefficient ;end
%             if rhomlt_coefficient>rhoscaling,rhomlt_coefficient=rhoscaling ;end
%             rsf_coefficient=1;
%             if Primal_coefficient>rhorsdltarget*rhorsdlratio*Dual_coefficient,rsf_coefficient=rhomlt_coefficient; end
%             if Dual_coefficient>(rhorsdlratio/rhorsdltarget)*Primal_coefficient,rsf_coefficient=rhomlt_coefficient; end
%             rhoC=rsf_coefficient*rhoC;
            %d_Z=d_Z/rsf_coefficient;
             %----------------------------------------------%
            if strcmp( verbose, 'brief')
                obj_val_z = objective(z, D{1});
                fprintf('Iter Z %d, Obj %3.3g, Diff %5.5g\n', i_z, obj_val_z, norm(z_diff(:),2)/ norm(z(:),2)) %  
            end
            if (norm(z_diff(:),2)/ norm(z(:),2) < tol)
                break;
            end
        end
        if  strcmp( verbose, 'all')
            display_func(iterate_fig, filter_fig, D{1}, z_hat, b, size_x, size_z_crop, psf_radius, i);
            fprintf('Sparse coding learning loop: %d\n\n', i)
        end
        
        iterations.obj_vals_d(i + 1) = obj_val_filter;
        iterations.obj_vals_z(i + 1) = obj_val_z;
        iterations.tim_vals(i + 1) = iterations.tim_vals(i) + t_kernel + t_vars;
        
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % %         if obj_val_min < obj_val_filter && obj_val_min < obj_val_z
% % % %             z = z_old;            
% % % %             d = d_old;
% % % %             iter = i-1;
% % % %             break;
% % % %         end
        %Termination
        if norm(z_diff(:),2)/ norm(z(:),2) < tol && norm(d_diff(:),2)/ norm(D{1}(:),2) < tol
            break;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
    end
    
    %Final estimate
    DZ = real(ifft2( sum(z_hat.* repmat(dup{1}, 1,1,1,n),3) ));
    
    d_res = circshift(D{1}, [psf_radius, psf_radius, 0] );
    d_res = d_res(1:psf_radius*2+1,1:psf_radius*2+1, :);
    z_res = z;  
    %obj_val = objective(z, D{1});
return;

function u = vec(v)

  u = v(:);

return
function [u_proj] = KernelConstraintProj( u, size_d, psf_radius)

    %Params
    k = size_d(end);
    ndim = length( size_d ) - 1;

    %Get support
    u_proj = circshift( u, [psf_radius, psf_radius, 0] ); %�����½ǵ�5x5�Ƶ������Ͻ�
    u_proj = u_proj(1:psf_radius*2+1,1:psf_radius*2+1,:); %ȡ���Ͻǵ�11x11
    
     %Normalize
 	u_norm = repmat( sum(sum(u_proj.^2, 1),2), [size(u_proj,1), size(u_proj,2), 1] );
    u_proj( u_norm >= 1 ) = u_proj( u_norm >= 1 ) ./ sqrt(u_norm( u_norm >= 1 ));%����d��һ��Ľ�������
    
    %Now shift back and pad again
    u_proj = padarray( u_proj, [size_d(1:end - 1) - (2*psf_radius+1), 0], 0, 'post');
    u_proj = circshift(u_proj, -[repmat(psf_radius, 1, ndim), 0]);
    
return;

function [zhat_mat, zhat_inv_mat] = precompute_H_hat_D(z_hat, size_z, rho)
% Computes the spectra for the inversion of all H_i

%Params
n = size_z(end);
ni = size_z(end);
k = size_z(end - 1);
ndim = length( size_z ) - 2;
ss = prod(size_z(1:ndim));

%Precompute spectra for H
zhat_mat = reshape( num2cell( permute( reshape(z_hat, [ss, k, ni] ), [3,2,1] ), [1 2] ), [1 ss]); %n * k * s

%Precompute the inverse matrices for each frequency
zhat_inv_mat = reshape( cellfun(@(A)(1/rho * eye(k) - 1/rho * A'*pinv(rho * eye(ni) + A * A')*A), zhat_mat, 'UniformOutput', false'), [1 ss]);

return;

function [dhat_flat, dhatTdhat_flat] = precompute_H_hat_Z(dhat, size_x )
% Computes the spectra for the inversion of all H_i

%Params
ndim = length( size_x ) - 1;
ss = prod(size_x(1:ndim));

%Precompute the dot products for each frequency
dhat_flat = reshape( dhat, ss, [] );
dhatTdhat_flat = sum(conj(dhat_flat).*dhat_flat,2);

return;

function d_hat = solve_conv_term_D(zhat_mat, zhat_inv_mat, d, B, rho, size_z )

    % Solves sum_j gamma_i/2 * || H_j d - xi_j ||_2^2
    % In our case: 1/2|| Zd - xi_1 ||_2^2 + rho * 1/2 * || d - xi_2||
    % with rho = gamma(2)/gamma(1)
    
    %Size
    n = size_z(end);
    ni=size_z(end);
    k = size_z(end - 1);
    ndim = length( size_z ) - 2;
    ss = prod(size_z(1:ndim));
    
    xi_hat_1_cell = num2cell( permute( reshape(B, ss, ni), [2,1] ), 1);
    xi_hat_2_cell = num2cell( permute( reshape(d, ss, k), [2,1] ), 1);
    
    %Invert
    x = cellfun(@(Sinv, A, b, c)(Sinv * (A' * b + rho * c)), zhat_inv_mat, zhat_mat,...
                                    xi_hat_1_cell, xi_hat_2_cell, 'UniformOutput', false);
    
    %Reshape to get back the new Dhat
    ss_size = size_z(1:ndim);
    d_hat = reshape( permute(cell2mat(x), [2,1]), [ss_size,k] );

return;

function z_hat = solve_conv_term_Z(dhatT, dhatTdhat, z, B, gammas, size_z )


    % Solves sum_j gamma_i/2 * || H_j z - xi_j ||_2^2
    % In our case: 1/2|| Dz - xi_1 ||_2^2 + rho * 1/2 * || z - xi_2||
    % with rho = gamma(2)/gamma(1)
    
    %Size
    ni = size_z(end);
    k = size_z(end - 1);
    ndim = length( size_z ) - 2;
    ss = prod(size_z(1:ndim));
    
    %Rho
    rho = gammas;
    
    %Compute b
    b = dhatT .* permute( repmat( reshape(B, ss, 1, ni), [1,k,1] ), [2,1,3] ) + rho .* permute( reshape(z, ss, k, ni), [2,1,3] );
    
    %Invert
    z_hat = 1/rho *b - 1/rho * repmat( ones([1,ss]) ./ ( rho * ones([1,ss]) + dhatTdhat.' ), [k,1,ni] ) .* dhatT .* repmat( sum(conj(dhatT).*b, 1), [k,1,1] );
    
    %Final transpose gives z_hat
    z_hat = reshape(permute(z_hat, [2,1,3]), size_z);

return;

function f_val = objectiveFunction(z, d, b, lambda_residual, lambda, psf_radius, size_z, size_x)
    
    %Params
    n = size_x(end);
    k = size_z(end-1);
    ndim = length( size_z ) - 2;
    Dz = zeros( size_x );
    all_dims = repmat(':,',1,ndim);
    
    
    Dz = real(ifft2( sum(fft2(z).* repmat(fft2(d), 1,1,1,n),3) ));
    f_z = lambda_residual * 1/2 * norm( reshape(  Dz(1 + psf_radius:end - psf_radius,1 + psf_radius:end - psf_radius,:) - b, [], 1) , 2 )^2; 
            
    g_z = lambda * sum( abs( z(:) ), 1 );
    
    %Function val
    f_val = f_z + g_z;
    
    
return;

function [] = display_func(iterate_fig, filter_fig, d, z_hat, b, size_x, size_z, psf_radius, iter)

    %Params
    n = size_x(end);
    k = size_z(end-1);
    ndim = length( size_z ) - 2;

    figure(iterate_fig); 
    Dz = zeros( size_x );
    all_dims = repmat(':,',1,ndim);
    
    Dz = real(ifft2( sum(z_hat.* repmat(fft2(d), 1,1,1,n),3) ));
    Dz = Dz(1 + psf_radius:end - psf_radius,1 + psf_radius:end - psf_radius,:);
    %Display some
    inds = repmat({6}, 1, ndim + 1);
    inds{1} = ':'; %Pick first two dims to show in 2D window
    inds{2} = ':';

    inds{end} = 1;
    subplot(3,2,1), imagesc(b(inds{:}));  axis image, colormap gray, title('Orig');
    subplot(3,2,2), imagesc(Dz(inds{:})); axis image, colormap gray; title(sprintf('Local iterate %d',iter));
    inds{end} = 2;
    subplot(3,2,3), imagesc(b(inds{:}));  axis image, colormap gray;
    subplot(3,2,4), imagesc(Dz(inds{:})); axis image, colormap gray;
    inds{end} = 3;
    subplot(3,2,5), imagesc(b(inds{:}));  axis image, colormap gray;
    subplot(3,2,6), imagesc(Dz(inds{:})); axis image, colormap gray;
    
    figure(filter_fig);
    sqr_k = ceil(sqrt(k));
    pd = 1;
    d_disp = zeros( sqr_k * [psf_radius*2+1 + pd, psf_radius*2+1 + pd] + [pd, pd]);
    inds = repmat({10}, 1, ndim + 1);
    inds{1} = ':'; %Pick first two dims to show in 2D window
    inds{2} = ':';
    for j = 0:k - 1
        inds{end} = j + 1;
        d_curr = circshift( d(inds{:}), [psf_radius, psf_radius] ); 
        d_curr = d_curr(1:psf_radius*2+1, 1:psf_radius*2+1);
        d_disp( floor(j/sqr_k) * (size(d_curr,1) + pd) + pd + (1:size(d_curr,1)) , mod(j,sqr_k) * (size(d_curr,2) + pd) + pd + (1:size(d_curr,2)) ) = d_curr;
    end
    imagesc(d_disp), colormap gray; axis image; colorbar; title(sprintf('Local filter iterate %d',iter));
    drawnow;        
return;
