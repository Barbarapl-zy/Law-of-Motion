
%% Full Simulation: Recursive Kalman Update + Burn-in + Eq(7) and Eq(8) Comparison

clear; clc;

%% Parameter Settings
N = 500;          % Monte Carlo simulation repetitions
T = 80;           % Final sample size (used for statistics)
T_burnin = 200;   % Burn-in periods
T_total = T + T_burnin;  % Total simulation length
M = 100;          % Number of forecasters
rho_true = 0.5;   % True AR(1) coefficient
rho_hat = 0.8;    % Perceived AR(1) coefficient by forecasters
omega = 0.5;      % Standard deviation of signal noise
h = 1;            % Forecast horizon

%% === Compute Steady-State Kalman Gain via Riccati Equation ===
Q = 1;
R = omega^2;
rho = rho_hat;

P = 1;
tol = 1e-10;
max_iter = 1000;

for iter = 1:max_iter
    P_new = rho^2 * (P - (P^2)/(P + R)) + Q;
    if abs(P_new - P) < tol
        break;
    end
    P = P_new;
end

K = P / (P + R);

%% Initialize Output Arrays
stat_eq7 = zeros(N,1);
stat_eq8 = zeros(N,1);

%% Monte Carlo Simulation Main Loop
for rep = 1:N
    % Step 1: Generate AR(1) time series with burn-in
    y = zeros(T_total,1);
    eps = randn(T_total,1);
    for t = 2:T_total
        y(t) = rho_true * y(t-1) + eps(t);
    end

    % Step 2: Recursive Kalman belief updates for each forecaster
    signals = zeros(M,T_total-1);
    yhat = zeros(M,T_total-1);
    for i = 1:M
        yhat(i,1) = 0;
        for t = 2:T_total
            signals(i,t-1) = y(t-1) + omega * randn();
            prior = rho_hat * yhat(i,t-2+1);
            innovation = signals(i,t-1) - prior;
            yhat(i,t-1) = prior + K * innovation;
        end
    end

    % Trim to post-burn-in values
    y = y((end-T+1):end);
    yhat = yhat(:, (end-T+1):end);
    signals = signals(:, (end-T+1):end);

    % Step 3: h-step ahead forecasts
    pred = zeros(M,T-h);
    for i = 1:M
        for t = 1:(T-h)
            pred(i,t) = rho_hat^h * yhat(i,t);
        end
    end

    % Step 4: Compute consensus forecasts and true values
    Ft_ytph = mean(pred,1)';
    Ftyt = mean(yhat,1)';
    yt = y(1:T-h);
    ytph = y((1+h):T);

    % Step 5: Compute statistic for Equation (7)
    cov1 = cov(Ft_ytph, yt);
    cov2 = cov(Ftyt, ytph);
    stat_eq7(rep) = cov1(1,2) / cov2(1,2);

    % Step 6: Compute statistic for Equation (8)
    cov3 = cov(Ft_ytph, Ft_ytph);
    cov4 = cov(ytph, Ft_ytph);
    cov5 = cov(yt, yt);
    cov6 = cov(ytph, yt);
    stat_eq8(rep) = (cov4(1,2)/cov3(1,1)) * (cov5(1,1)/cov6(1,2));
end

%% Display Mean Results
fprintf("Mean of Equation (7) statistics: %.4f\n", mean(stat_eq7));
fprintf("Mean of Equation (8) statistics: %.4f\n", mean(stat_eq8));
