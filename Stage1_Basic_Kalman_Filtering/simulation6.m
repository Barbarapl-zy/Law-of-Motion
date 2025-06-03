%% Full Simulation: Recursive Kalman Update + Burn-in + Eq(7) and Eq(8) Bias Estimation
clear; clc;

%% 1. Parameter Settings
N = 500; 
T = 80; 
T_burnin = 200; 
T_total = T + T_burnin;
M = 100; 
rho_true = 0.5; 
rho_hat = 0.8; 
omega = 0.5; 
h = 1;

%% 2. Steady-State Kalman Gain via Riccati Iteration (CORRECTED)
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

%% 4. Monte Carlo Simulation Loop
for rep = 1:N
    %% 4.1 Generate AR(1) Series (CORRECTED: proper initialization)
    eps = randn(T_total, 1);
    y = zeros(T_total, 1);
    y(1) = randn(); % Start from random initial state
    for t = 2:T_total
        y(t) = rho_true * y(t-1) + eps(t);
    end

    %% 4.2 Kalman Filtering (CORRECTED: time indexing and signal alignment)
    signals = zeros(M, T_total);
    yhat_prior = zeros(M, T_total);
    yhat_post = zeros(M, T_total);
    
    % Generate signals for time t=1 to T_total (observed at each time)
    for i = 1:M
        for t = 1:T_total
            signals(i, t) = y(t) + omega * randn(); % Observe CURRENT state y(t)
        end
    end
    
    % Initialize KF
    yhat_post(:, 1) = 0;  % Initial posterior estimate
    
    % Kalman filter recursion
    for i = 1:M
        for t = 2:T_total
            % 1. Predict step (prior)
            yhat_prior(i, t) = rho_hat * yhat_post(i, t-1);
            
            % 2. Update step (posterior)
            innovation = signals(i, t) - yhat_prior(i, t);
            yhat_post(i, t) = yhat_prior(i, t) + K * innovation;
        end
    end

    %% 4.3 Trim Burn-in (keep last T points)
    y_effective = y((end-T+1):end);
    yhat_post_eff = yhat_post(:, (end-T+1):end);
    signals_eff = signals(:, (end-T+1):end);

    %% 4.4 Generate h-step Forecasts (CORRECTED: time indexing)
    yhat_h_step = zeros(M, T - h);
    for i = 1:M
        for tau = 1:(T - h)  % tau is index in effective window
            % Forecast made at time (tau) for time (tau + h)
            yhat_h_step(i, tau) = (rho_hat)^h * yhat_post_eff(i, tau);
        end
    end

    %% 4.5 Consensus Forecasts and Truth (CORRECTED: alignment)
    % Average over analysts (M dimension)
    avg_forecast_h = mean(yhat_h_step, 1)';  % (T-h) x 1
    avg_posterior = mean(yhat_post_eff, 1)'; % T x 1
    
    % Extract true values (aligned with forecasts)
    y_true_t = y_effective(1:T-h);          % True state at time t
    y_true_tph = y_effective(1+h:T);        % True state at time t+h
    
    % Posterior estimates at time t (used for forecast)
    avg_posterior_t = avg_posterior(1:T-h); % Posterior at time t

    %% 4.6 Statistic (7) (CORRECTED: covariance inputs)
    % Numerator: Cov( forecast_{t->t+h}, y_t )
    % Denominator: Cov( x_t, y_{t+h} )
    cov_num = cov(avg_forecast_h, y_true_t);
    cov_den = cov(avg_posterior_t, y_true_tph);
    stat_eq7(rep) = cov_num(1,2) / cov_den(1,2);

    %% 4.7 Statistic (8) (CORRECTED: covariance inputs)
    % Term1: Cov(y_{t+h}, forecast_{t->t+h}) / Var(forecast_{t->t+h})
    % Term2: Var(y_t) / Cov(y_{t+h}, y_t)
    cov_yf = cov(y_true_tph, avg_forecast_h);
    cov_ff = cov(avg_forecast_h, avg_forecast_h);
    term1 = cov_yf(1,2) / cov_ff(1,1);
    
    cov_yy = cov(y_true_t, y_true_t);
    cov_yyp = cov(y_true_tph, y_true_t);
    term2 = cov_yy(1,1) / cov_yyp(1,2);
    
    stat_eq8(rep) = term1 * term2;

    %% 4.8 Bias
    bias_eq7(rep) = stat_eq7(rep) - rho_hat;
    bias_eq8(rep) = stat_eq8(rep) - rho_hat;
end

%% Results Display
fprintf("=== Statistical Power ===\n");
fprintf("Mean of Eq(7): %.4f\n", mean(stat_eq7));
fprintf("Mean of Eq(8): %.4f\n", mean(stat_eq8));
fprintf("\n=== Bias Estimation Performance ===\n");
fprintf("Eq(7): Mean Bias = %.4f, Std Dev = %.4f\n", mean(bias_eq7), std(bias_eq7));
fprintf("Eq(8): Mean Bias = %.4f, Std Dev = %.4f\n", mean(bias_eq8), std(bias_eq8));
fprintf("\nRatio of Mean(Eq7) to Mean(Eq8): %.4f\n", mean(stat_eq7) / mean(stat_eq8));

%% Visualization
figure;
histogram(bias_eq7, 'Normalization', 'pdf', 'FaceAlpha', 0.6); hold on;
histogram(bias_eq8, 'Normalization', 'pdf', 'FaceAlpha', 0.6);
legend('Bias from Eq(7)', 'Bias from Eq(8)');
title('Distribution of Perceived Rho Bias Estimates');
xlabel('Bias (estimated rho\_hat - true rho\_hat)');
ylabel('Density');