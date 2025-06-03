
%% simulation10_AR2_analysis_corrected.m
% Simulates a setting where the true DGP is AR(2), the forecaster knows it and uses Kalman Filter,
% but the analyst mis-specifies the model as AR(1) and tests forecast rationality using stat(7) and stat(8)

clear; clc;

%% 1. Simulation Parameters
N = 500;            % Monte Carlo replications
T = 80;             % Sample length (after burn-in)
T_burnin = 200;     % Burn-in period
T_total = T + T_burnin;
M = 100;            % Number of forecasters
omega = 0.5;        % Std. dev. of signal noise
h = 1;              % Forecast horizon

%% 2. AR(2) Parameters (True DGP)
rho1 = 1.3;
rho2 = -0.4;
A = [rho1, rho2; 1, 0];
H = [1, 0];                 % Observe only y_t
Q = [1 0; 0 0];             % Shock in y-equation only
R = omega^2;

%% 3. Result Storage
stat_eq7_AR2 = zeros(N,1);
stat_eq8_AR2 = zeros(N,1);

%% 4. Monte Carlo Loop
for rep = 1:N
    % 4.1 Generate AR(2) process
    eps = randn(T_total, 1);
    y = zeros(T_total, 1);
    y(1:2) = randn(2,1);
    for t = 3:T_total
        y(t) = rho1 * y(t-1) + rho2 * y(t-2) + eps(t);
    end

    % 4.2 Generate noisy signals
    signals = zeros(M, T_total);
    for i = 1:M
        for t = 1:T_total
            signals(i, t) = y(t) + omega * randn();
        end
    end

    % 4.3 Vector Kalman Filter
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

    % 4.4 Forecast and posterior extraction
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

    %% 4.5 Analyst computes stat(7) and stat(8) under AR(1) misbelief

    % True values
    avg_forecast_h = mean(yhat_h_step, 1)';
    avg_posterior_t = mean(yhat_post_scalar, 1)';
    y_true_t = y_effective(1:T-h);
    y_true_tph = y_effective(1+h:end);

    % Eq(7): structure-agnostic
    cov_num = mean((avg_forecast_h - mean(avg_forecast_h)) .* ...
                   (y_true_t - mean(y_true_t)));
    cov_den = mean((avg_posterior_t - mean(avg_posterior_t)) .* ...
                   (y_true_tph - mean(y_true_tph)));
    stat_eq7_AR2(rep) = cov_num / cov_den;

    % Eq(8): analyst wrongly assumes forecast = rho_hat^h * posterior
    rho_hat = 0.8;  % analyst's belief
    forecast_analyst = rho_hat^h * avg_posterior_t;

    cov_yf = mean((avg_posterior_t - mean(avg_posterior_t)) .* ...
                  (forecast_analyst - mean(forecast_analyst)));
    var_f = mean((avg_posterior_t - mean(avg_posterior_t)).^2);
    term1 = cov_yf / var_f;

    var_y = mean((y_true_t - mean(y_true_t)).^2);
    cov_yyp = mean((y_true_tph - mean(y_true_tph)) .* ...
                   (y_true_t - mean(y_true_t)));
    term2 = var_y / cov_yyp;

    stat_eq8_AR2(rep) = term1 * term2;
end

%% 5. Report Results
fprintf("\n=== AR(2) DGP with Analyst Assuming AR(1) ===\n");
fprintf("Mean of stat(7): %.4f\n", mean(stat_eq7_AR2));
fprintf("Mean of stat(8): %.4f\n", mean(stat_eq8_AR2));
fprintf("Ratio stat(7) / stat(8): %.4f\n", mean(stat_eq7_AR2) / mean(stat_eq8_AR2));

%% 6. Plot KDE
figure;
ksdensity(stat_eq7_AR2); hold on;
ksdensity(stat_eq8_AR2);
xline(1, '--k');
legend('stat(7)', 'stat(8)', 'Location', 'best');
title('Distribution of stat(7) and stat(8) under AR(2) True Model');
xlabel('Statistic Value');
ylabel('Density');
grid on;
