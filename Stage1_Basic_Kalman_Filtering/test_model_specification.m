%% 模型设定检验函数
function test_model_specification()
    % 参数设置
    N = 500; T = 80; T_burnin = 200; T_total = T + T_burnin;
    M = 100; omega = 0.5; h = 1;
    
    % 创建不同设定场景
    scenarios = {
        struct('name', 'Correctly Specified', 'rho_true', 0.5, 'rho_hat', 0.5),...
        struct('name', 'Misspecified (Optimistic)', 'rho_true', 0.5, 'rho_hat', 0.8),...
        struct('name', 'Misspecified (Pessimistic)', 'rho_true', 0.5, 'rho_hat', 0.3)
    };
    
    results = cell(length(scenarios), 1);
    
    for s = 1:length(scenarios)
        scen = scenarios{s};
        fprintf('\n=== Testing Scenario: %s ===\n', scen.name);
        fprintf('True ρ = %.2f, Perceived ρ = %.2f\n', scen.rho_true, scen.rho_hat);
        
        % 运行蒙特卡洛模拟
        [bias_eq7, bias_eq8] = run_monte_carlo_simulation(...
            N, T, T_burnin, M, scen.rho_true, scen.rho_hat, omega, h);
        
        % 存储结果
        results{s} = struct(...
            'name', scen.name,...
            'bias_eq7', bias_eq7,...
            'bias_eq8', bias_eq8,...
            'mean_bias7', mean(bias_eq7),...
            'mean_bias8', mean(bias_eq8));
        
        % 显示结果
        fprintf('Eq7 Bias: Mean = %.4f, Std = %.4f\n', mean(bias_eq7), std(bias_eq7));
        fprintf('Eq8 Bias: Mean = %.4f, Std = %.4f\n', mean(bias_eq8), std(bias_eq8));
    end
    
    %% 可视化结果
    figure('Position', [100, 100, 1200, 500]);
    
    % 偏差分布图
    subplot(1,2,1);
    hold on;
    colors = lines(length(scenarios));
    for s = 1:length(scenarios)
        histogram(results{s}.bias_eq7, 'BinWidth', 0.02, 'FaceAlpha', 0.6,...
                 'DisplayName', [results{s}.name ' (Eq7)'], 'EdgeColor', 'none');
        histogram(results{s}.bias_eq8, 'BinWidth', 0.02, 'FaceAlpha', 0.6,...
                 'DisplayName', [results{s}.name ' (Eq8)'], 'EdgeColor', 'none',...
                 'FaceColor', colors(s,:) * 0.7);
    end
    xline(0, '--', 'Reference Line', 'LineWidth', 1.5);
    title('Bias Distribution Across Model Specifications');
    xlabel('Bias (estimated - perceived ρ)');
    ylabel('Frequency');
    legend('show');
    grid on;
    
    % 平均偏差比较图
    subplot(1,2,2);
    mean_biases = cell2mat(cellfun(@(r) [r.mean_bias7; r.mean_bias8], results, 'UniformOutput', false));
    bar(mean_biases', 'grouped');
    set(gca, 'XTickLabel', {scenarios.name});
    title('Average Bias by Model Specification');
    ylabel('Mean Bias');
    legend('Eq7 Bias', 'Eq8 Bias');
    grid on;
end

%% 蒙特卡洛模拟函数
function [bias_eq7, bias_eq8] = run_monte_carlo_simulation(...
    N, T, T_burnin, M, rho_true, rho_hat, omega, h)
    
    T_total = T + T_burnin;
    
    % 计算稳态卡尔曼增益
    Q = 1; R = omega^2;
    P_inf = 1;
    tol = 1e-10; max_iter = 1000;
    for iter = 1:max_iter
        P_next = rho_hat^2 * (P_inf - (P_inf^2)/(P_inf + R)) + Q;
        if abs(P_next - P_inf) < tol, break; end
        P_inf = P_next;
    end
    K = P_inf / (P_inf + R);

    % 初始化结果数组
    bias_eq7 = zeros(N,1);
    bias_eq8 = zeros(N,1);
    
    for rep = 1:N
        % 生成AR(1)序列
        eps = randn(T_total, 1);
        y = zeros(T_total, 1);
        y(1) = randn();
        for t = 2:T_total
            y(t) = rho_true * y(t-1) + eps(t);
        end
        
        % 卡尔曼滤波（修正后的顺序）
        signals = zeros(M, T_total);
        yhat_prior = zeros(M, T_total);
        yhat_post = zeros(M, T_total);
        
        % 生成含噪声信号
        for i = 1:M
            signals(i, :) = y + omega * randn(size(y));
        end
        
        % 初始化
        yhat_post(:, 1) = 0;
        
        % 滤波循环（先预测后更新）
        for i = 1:M
            for t = 2:T_total
                % 预测步骤 (t|t-1)
                yhat_prior(i, t) = rho_hat * yhat_post(i, t-1);
                
                % 更新步骤 (t|t)
                innovation = signals(i, t) - yhat_prior(i, t);
                yhat_post(i, t) = yhat_prior(i, t) + K * innovation;
            end
        end
        
        % 去除预热期
        y_effective = y(end-T+1:end);
        yhat_post_eff = yhat_post(:, end-T+1:end);
        
        % 生成h步预测
        yhat_h_step = zeros(M, T - h);
        for i = 1:M
            for tau = 1:(T - h)
                yhat_h_step(i, tau) = rho_hat^h * yhat_post_eff(i, tau);
            end
        end
        
        % 计算共识预测
        avg_forecast_h = mean(yhat_h_step, 1)';
        avg_posterior = mean(yhat_post_eff, 1)';
        avg_posterior_t = avg_posterior(1:T-h);
        
        % 对齐真实值
        y_true_t = y_effective(1:T-h);
        y_true_tph = y_effective(h+1:T);
        
        % 计算统计量
        cov_num = cov(avg_forecast_h, y_true_t);
        cov_den = cov(avg_posterior_t, y_true_tph);
        stat_eq7 = cov_num(1,2) / cov_den(1,2);
        
        cov_yf = cov(y_true_tph, avg_forecast_h);
        cov_ff = cov(avg_forecast_h);
        cov_yy = cov(y_true_t);
        cov_yyp = cov(y_true_tph, y_true_t);
        stat_eq8 = (cov_yf(1,2)/cov_ff(1,1)) * (cov_yy(1,1)/cov_yyp(1,2));
        
        % 计算偏差
        bias_eq7(rep) = stat_eq7 - rho_hat;
        bias_eq8(rep) = stat_eq8 - rho_hat;
    end
end