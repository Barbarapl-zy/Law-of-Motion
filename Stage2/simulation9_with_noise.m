%% Full Simulation
clear; clc;

%% 1. Parameter Settings
N = 500; 
T = 2000; 
T_burnin = 1000; 
T_total = T + T_burnin;
M = 100; 
rho_true = 0.5; 
rho_hat = 0.8; 
omega = 0.2; 
h = 1;

%% 2. Steady-State Kalman Gain via Riccati Iteration
Q = 1; 
R = omega^2; 
rho = rho_hat;
P_inf = 1;  % Initial covariance
tol = 1e-10; 
max_iter = 1000;
for iter = 1:max_iter
    % Correct Riccati iteration (fully expanded)
    P_next = rho^2 * (P_inf - (P_inf^2)/(P_inf + R)) + Q;
    if abs(P_next - P_inf) < tol
        break;
    end
    P_inf = P_next;
end
K = P_inf / (P_inf + R); % Steady-state Kalman gain

%% 3. Initialize Result Arrays
stat_eq7 = zeros(N,1); stat_eq8 = zeros(N,1);
bias_eq7 = zeros(N,1); bias_eq8 = zeros(N,1);

stat_eq7_noise = zeros(N,1);
stat_eq8_noise = zeros(N,1);

%% 4. Monte Carlo Simulation Loop
for rep = 1:N
    %% 4.1 Generate AR(1) Series
    eps = randn(T_total, 1);
    y = zeros(T_total, 1);
    % Start from steady-state distribution
    y(1) = randn() * sqrt(1/(1 - rho_true^2)); 
    for t = 2:T_total
        y(t) = rho_true * y(t-1) + eps(t);
    end

    %% 4.2 Kalman Filtering
    signals = zeros(M, T_total);
    yhat_prior = zeros(M, T_total);
    yhat_post = zeros(M, T_total);
    
    % Generate signals for time t=1 to T_total
    for i = 1:M
        for t = 1:T_total
            signals(i, t) = y(t) + omega * randn();
        end
    end
        
    % Kalman filter recursion
    for i = 1:M
        % 初始化时点 t=1：只可算 posterior，不能预测 t=0
        yhat_post(i, 1) = mean(signals(:, 1));  % 初始 posterior
        % loop
        for t = 2:T_total
            % 1. Predict step (prior at time t based on previous posterior)
            yhat_prior(i, t) = rho_hat * yhat_post(i, t-1);  % y_{t|t-1}
    
            % 2. Update step using signal at time t
            innovation = signals(i, t) - yhat_prior(i, t);  % s_t - y_{t|t-1}
            yhat_post(i, t) = yhat_prior(i, t) + K * innovation;  % y_{t|t}
        end
    end

    %% 4.3 Trim Burn-in (keep last T points)
    start_idx = T_total - T + 1;
    y_effective = y(start_idx:end);
    yhat_post_eff = yhat_post(:, start_idx:end);

    %% 4.4 Generate h-step Forecasts
    % yhat_h_step = zeros(M, T - h);
    % for i = 1:M
    %     for tau = 1:(T - h)
    %         % Forecast made at time tau for time tau+h
    %         yhat_h_step(i, tau) = (rho_hat)^h * yhat_post_eff(i, tau);
    %     end
    % end
    yhat_h_step = (rho_hat)^h * yhat_post_eff;

    %% 4.5 Consensus Forecasts and Truth
    % Average over analysts
    avg_forecast_h = mean(yhat_h_step(:,1:T-h), 1)';  % (T-h) x 1
    avg_posterior = mean(yhat_post_eff(:,1:T-h), 1)'; % T x 1
    
    % Extract true values (aligned with forecasts)
    % Forecast made at time t for time t+h
    y_true_t = y_effective(1:T-h);          % True state at forecast time t
    y_true_tph = y_effective(1+h:end);      % True state at target time t+h
    
    % Posterior estimates at forecast time t
    avg_posterior_t = avg_posterior(1:T-h); % Posterior at time t

    %% 4.6 Statistic (7)
    % Numerator: Cov( forecast_{t->t+h}, y_t )
    % Denominator: Cov( x_t, y_{t+h} )
    
    % Custom covariance calculation to avoid matrix output issues
    cov_num = mean((avg_forecast_h - mean(avg_forecast_h)) .* (y_true_t - mean(y_true_t)));
    cov_den = mean((avg_posterior_t - mean(avg_posterior_t)) .* (y_true_tph - mean(y_true_tph)));
    stat_eq7(rep) = cov_num / cov_den;

    %% 4.7 Statistic (8)
    % Term1: Cov(forecast_{t->t+h}, forecast_{t->t}) / Var(forecast_{t->t})
    % Term2: Var(y_t) / Cov(y_{t+h}, y_t)
    
    cov_yf = mean((avg_posterior_t - mean(avg_posterior_t)) .* (avg_forecast_h - mean(avg_forecast_h)));
    var_f = mean((avg_posterior_t - mean(avg_posterior_t)).^2);
    term1 = cov_yf / var_f;
    
    var_y = mean((y_true_t - mean(y_true_t)).^2);
    cov_yyp = mean((y_true_tph - mean(y_true_tph)) .* (y_true_t - mean(y_true_t)));
    term2 = var_y / cov_yyp;
    
    stat_eq8(rep) = term1 * term2;


    %% 4.8 Noisy Observed Aggregates  (New)
    noise_std = 0.05; 
    
    % Add noise to aggregate-level statistics
    avg_forecast_h_obs = avg_forecast_h + noise_std * randn(size(avg_forecast_h));
    avg_posterior_obs = avg_posterior + noise_std * randn(size(avg_posterior));
    avg_posterior_t_obs = avg_posterior_obs(1:T-h);
    
    % Compute Eq(7) with noisy aggregate data
    cov_num_noise = mean((avg_forecast_h_obs - mean(avg_forecast_h_obs)) .* ...
                         (y_true_t - mean(y_true_t)));
    cov_den_noise = mean((avg_posterior_t_obs - mean(avg_posterior_t_obs)) .* ...
                         (y_true_tph - mean(y_true_tph)));
    stat_eq7_noise(rep) = cov_num_noise / cov_den_noise;
    
    % Compute Eq(8) with noisy aggregate data
    cov_yf_noise = mean((avg_posterior_t_obs - mean(avg_posterior_t_obs)) .* ...
                        (avg_forecast_h_obs - mean(avg_forecast_h_obs)));
    var_f_noise = mean((avg_posterior_t_obs - mean(avg_posterior_t_obs)).^2);
    term1_noise = cov_yf_noise / var_f_noise;
    
    var_y = mean((y_true_t - mean(y_true_t)).^2);
    cov_yyp = mean((y_true_tph - mean(y_true_tph)) .* ...
                   (y_true_t - mean(y_true_t)));
    term2 = var_y / cov_yyp;
    
    stat_eq8_noise(rep) = term1_noise * term2;

end

%% Results Display
fprintf("=== Statistical Power ===\n");
fprintf("Mean of Eq(7): %.4f\n", mean(stat_eq7));
fprintf("Mean of Eq(8): %.4f\n", mean(stat_eq8));
fprintf("\nRatio of Mean(Eq7) to Mean(Eq8): %.4f\n", mean(stat_eq7) / mean(stat_eq8));

fprintf("=== With Aggregate Noise (std = %.2f) ===\n", noise_std);
fprintf("Mean of Noisy Eq(7): %.4f\n", mean(stat_eq7_noise));
fprintf("Mean of Noisy Eq(8): %.4f\n", mean(stat_eq8_noise));
fprintf("Ratio of Noisy Eq(7) to Noisy Eq(8): %.4f\n\n", ...
         mean(stat_eq7_noise) / mean(stat_eq8_noise));



%% Visualization
figure;
% subplot(2,1,1);
histogram(stat_eq7, 'BinWidth', 0.05, 'Normalization', 'pdf', 'FaceAlpha', 0.6); hold on;
histogram(stat_eq8, 'BinWidth', 0.05, 'Normalization', 'pdf', 'FaceAlpha', 0.6);
xline(rho_hat, 'r--', 'LineWidth', 2, 'DisplayName', 'Perceived \rho');
legend('Statistic Eq(7)', 'Statistic Eq(8)', 'Perceived \rho');
title('Distribution of Statistics');
xlabel('Statistic Value');
ylabel('Density');


