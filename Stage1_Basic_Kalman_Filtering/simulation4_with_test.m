%% Full Simulation: Recursive Kalman Update + Burn-in + Eq(7) and Eq(8) Bias Estimation
clear; clc;

%% 1. Parameter Settings
N = 500;            % Number of Monte Carlo simulations
T = 80;             % Final sample size (after burn-in)
T_burnin = 200;     % Burn-in period
T_total = T + T_burnin;
M = 100;            % Number of forecasters
rho_true = 0.5;     % True AR(1) coefficient
rho_hat = 0.8;      % Perceived AR(1) coefficient by forecasters
omega = 0.5;        % Standard deviation of signal noise
h = 1;              % Forecast horizon

%% 2. Steady-State Kalman Gain via Riccati Iteration
Q = 1;              % Process noise variance
R = omega^2;        % Signal noise variance
rho = rho_hat;      % Used for computing steady state under perceived model

P = 1;              % Initial guess for prediction error variance
tol = 1e-10;        % Convergence tolerance
max_iter = 1000;

for iter = 1:max_iter
    P_new = rho^2 * (P - (P^2)/(P + R)) + Q;
    if abs(P_new - P) < tol
        break;
    end
    P = P_new;
end
K = P / (P + R);     % Steady-state Kalman gain

%% 3. Initialize Result Arrays
stat_eq7 = zeros(N,1);
stat_eq8 = zeros(N,1);
bias_eq7 = zeros(N,1);
bias_eq8 = zeros(N,1);

%% 4. Monte Carlo Simulation Loop
for rep = 1:N

    %% 4.1 Generate AR(1) Series
    y = zeros(T_total,1);
    eps = randn(T_total,1);
    for t = 2:T_total
        y(t) = rho_true * y(t-1) + eps(t);
    end

    %% 4.2 Kalman Filtering (Separate prior and posterior)
    signals = zeros(M, T_total-1);        % Noisy signals observed by forecasters
    yhat_prior = zeros(M, T_total);       % Prior estimates (t|t-1)
    yhat_post  = zeros(M, T_total);       % Posterior estimates (t|t)

    % Generate signals: each forecaster sees y(t-1) with noise
    for i = 1:M
        for t = 2:T_total
            signals(i, t-1) = y(t-1) + omega * randn();
        end
    end

    % Initialize posterior at t = 1
    yhat_prior(:, 1) = 0;       % we set t=1 prior to 0
    yhat_post(:, 1) = 0;

    % Kalman filtering loop
    for i = 1:M
        for t = 1:(T_total-1)   % stop at T_total - 1 to compute t+1|t safely
            % Step 1: correction (obtain posterior from prior)
            innovation = signals(i, t) - yhat_prior(i, t);           % signal = y(t) + noise, observed at t
            yhat_post(i, t) = yhat_prior(i, t) + K * innovation;     % t|t ← t|t-1 + correction

            % Step 2: prediction (obtain t+1|t from t|t)
            yhat_prior(i, t+1) = rho_hat * yhat_post(i, t);          % t+1|t ← rho * t|t
        end
    end

    %% 4.3 Trim Burn-in
    y = y((end-T+1):end); 
    yhat_post = yhat_post(:, (end-T+1):end);     % keep posterior only
    yhat_prior = yhat_prior(:, (end-T+1):end);   % optional if needed later
    signals = signals(:, (end-T+1):end);
    
    %% 4.4 Generate h-step Forecasts (based on posterior)
    yhat_h_step = zeros(M, T-h);                           % h-step forecasts
    for i = 1:M
        for t = 1:(T-h)
            yhat_h_step(i,t) = rho_hat^h * yhat_post(i,t); % y_{t+h|t} = rho^h * y_{t|t}
        end
    end
    
    %% 4.5 Consensus Forecasts and True Values
    avg_forecast_h = mean(yhat_h_step,1)';                 % E[y_{t+h|t}]
    avg_posterior = mean(yhat_post,1)';                    % E[y_{t|t}]
    avg_posterior = avg_posterior(1:T-h);                  % align dimension
    
    y_true_t = y(1:T-h);                                   % actual y_t
    y_true_tph = y((1+h):T);                               % actual y_{t+h}


    %% 4.6 Statistic (7)
    cov1 = cov(avg_forecast_h, y_true_t);
    cov2 = cov(avg_posterior, y_true_tph);
    stat_eq7(rep) = cov1(1,2) / cov2(1,2);
    
    %% 4.7 Statistic (8)
    cov3 = cov(avg_forecast_h, avg_forecast_h);
    cov4 = cov(y_true_tph, avg_forecast_h);
    cov5 = cov(y_true_t, y_true_t);
    cov6 = cov(y_true_tph, y_true_t);
    stat_eq8(rep) = (cov4(1,2)/cov3(1,1)) * (cov5(1,1)/cov6(1,2));

    %% 4.8 Bias Relative to Perceived Rho
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
    xlabel('Bias (estimated rho_hat - true rho_hat)');
    ylabel('Density');
