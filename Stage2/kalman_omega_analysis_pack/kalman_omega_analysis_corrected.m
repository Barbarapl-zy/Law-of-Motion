
%% kalman_omega_analysis_corrected.m
% Omega-sweep version of simulation10_AR2_analysis_corrected.m

clear; clc;

%% 1. Simulation Parameters
N = 500;            
T = 80;             
T_burnin = 200;     
T_total = T + T_burnin;
M = 100;            
h = 1;              
omega_list = [0.1, 0.3, 0.5, 1.0, 2.0];

%% 2. AR(2) Parameters (True DGP)
rho1 = 1.3;
rho2 = -0.4;
A = [rho1, rho2; 1, 0];
H = [1, 0];                 
Q = [1 0; 0 0];             

%% 3. Result Storage
stat7_matrix = zeros(length(omega_list), N);
stat8_matrix = zeros(length(omega_list), N);

%% 4. Loop over omega values
for w = 1:length(omega_list)
    omega = omega_list(w);
    R = omega^2;

    for rep = 1:N
        eps = randn(T_total, 1);
        y = zeros(T_total, 1);
        y(1:2) = randn(2,1);
        for t = 3:T_total
            y(t) = rho1 * y(t-1) + rho2 * y(t-2) + eps(t);
        end

        signals = zeros(M, T_total);
        for i = 1:M
            for t = 1:T_total
                signals(i, t) = y(t) + omega * randn();
            end
        end

        xhat_post = zeros(M, 2, T_total);
        P = repmat(eye(2), [1 1 M]);
        for i = 1:M
            xhat_post(i,:,2) = [signals(i,2), signals(i,1)];
        end

        for t = 3:T_total
            for i = 1:M
                x_prior = A * squeeze(xhat_post(i,:,t-1))';
                P_prior = A * P(:,:,i) * A' + Q;
                S = H * P_prior * H' + R;
                K = (P_prior * H') / S;
                innovation = signals(i,t) - H * x_prior;
                x_post = x_prior + K * innovation;
                P_post = (eye(2) - K * H) * P_prior;
                xhat_post(i,:,t) = x_post';
                P(:,:,i) = P_post;
            end
        end

        Ah = A^h;
        xhat_post_eff = xhat_post(:,:,T_burnin+1:end);
        y_effective = y(T_burnin+1:end);

        yhat_h_step = zeros(M, T - h);
        yhat_post_scalar = zeros(M, T - h);

        for i = 1:M
            for t = 1:(T - h)
                x_tt = squeeze(xhat_post_eff(i,:,t))';
                yhat_h_step(i,t) = H * Ah * x_tt;
                yhat_post_scalar(i,t) = H * x_tt;
            end
        end

        avg_forecast_h = mean(yhat_h_step, 1)';
        avg_posterior_t = mean(yhat_post_scalar, 1)';
        y_true_t = y_effective(1:T-h);
        y_true_tph = y_effective(1+h:end);

        cov_num = mean((avg_forecast_h - mean(avg_forecast_h)) .* ...
                       (y_true_t - mean(y_true_t)));
        cov_den = mean((avg_posterior_t - mean(avg_posterior_t)) .* ...
                       (y_true_tph - mean(y_true_tph)));
        stat7_matrix(w, rep) = cov_num / cov_den;

        rho_hat = 0.8;
        forecast_analyst = rho_hat^h * avg_posterior_t;

        cov_yf = mean((avg_posterior_t - mean(avg_posterior_t)) .* ...
                      (forecast_analyst - mean(forecast_analyst)));
        var_f = mean((avg_posterior_t - mean(avg_posterior_t)).^2);
        term1 = cov_yf / var_f;

        var_y = mean((y_true_t - mean(y_true_t)).^2);
        cov_yyp = mean((y_true_tph - mean(y_true_tph)) .* ...
                       (y_true_t - mean(y_true_t)));
        term2 = var_y / cov_yyp;

        stat8_matrix(w, rep) = term1 * term2;
    end
end

save('omega_analysis_corrected_results.mat', 'omega_list', 'stat7_matrix', 'stat8_matrix');
