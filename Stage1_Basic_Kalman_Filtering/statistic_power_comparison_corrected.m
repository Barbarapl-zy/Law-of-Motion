clear; clc;

%% 参数设置
N = 500;          % Monte Carlo模拟次数
T = 80;           % 时间序列长度
M = 100;          % 预测者数量
rho_true = 0.5;   % 真实AR(1)系数
rho_hat = 0.8;    % 预测者感知的AR(1)
omega = 0.5;      % 信号噪声标准差
h = 1;            % 预测期数

stat_eq7 = zeros(N,1);
stat_eq8 = zeros(N,1);

%% Monte Carlo主循环
for rep = 1:N
    % Step 1: 生成AR(1)序列
    y = zeros(T,1);
    eps = randn(T,1);
    for t = 2:T
        y(t) = rho_true * y(t-1) + eps(t);
    end

    % Step 2: 模拟预测者观测带噪signal并进行Kalman更新
    signals = zeros(M,T-1);
    yhat = zeros(M,T-1);
    for i = 1:M
        for t = 2:T
            signals(i,t-1) = y(t-1) + omega * randn();
            K = 1 / (1 + omega^2);  % Kalman gain
            yhat(i,t-1) = K * signals(i,t-1);
        end
    end

    % Step 3: 预测者构造未来预测
    pred = zeros(M,T-h);
    for i = 1:M
        for t = 1:(T-h)
            pred(i,t) = rho_hat^h * yhat(i,t);
        end
    end

    % Step 4: 共识预测与实际数据(revise)
    Ft_y_tph = mean(pred,1)';                  % E[y_{t+h} | info_t]
    Ft_y_t = rho_hat^h * mean(yhat(:,1:T-h),1)'; % E[y_t | info_t]
    yt = y(1:T-h);                             % y_t
    ytph = y((1+h):T);                         % y_{t+h}

    % Step 5: 统计量 (7) 
    cov1 = cov(Ft_y_tph, yt);
    cov2 = cov(ytph, Ft_y_t);
    stat_eq7(rep) = cov1(1,2) / cov2(1,2);

    % Step 6: 统计量 (8)
    covA = cov(Ft_y_tph, Ft_y_t);  % cov(F_t y_{t+h}, F_t y_t)
    varF = var(Ft_y_t);
    covB = cov(ytph, yt);
    stat_eq8(rep) = (covA(1,2)/varF) * (var(yt)/covB(1,2));
end

%% Bootstrap检验与Power分析
B = 1000;       % Bootstrap次数
alpha = 0.05;   % 显著性水平

boot_stat_eq7 = zeros(B,1);
boot_stat_eq8 = zeros(B,1);

for b = 1:B
    idx = randi(N, N, 1); % 有放回采样
    boot_stat_eq7(b) = mean(stat_eq7(idx));
    boot_stat_eq8(b) = mean(stat_eq8(idx));
end

boot_stat_eq7 = sort(boot_stat_eq7);
boot_stat_eq8 = sort(boot_stat_eq8);

ci7_low = boot_stat_eq7(floor(B * alpha/2));
ci7_high = boot_stat_eq7(ceil(B * (1 - alpha/2)));

ci8_low = boot_stat_eq8(floor(B * alpha/2));
ci8_high = boot_stat_eq8(ceil(B * (1 - alpha/2)));

reject_eq7 = (stat_eq7 < ci7_low) | (stat_eq7 > ci7_high);
reject_eq8 = (stat_eq8 < ci8_low) | (stat_eq8 > ci8_high);
power_eq7 = mean(reject_eq7);
power_eq8 = mean(reject_eq8);


%% 
%% 输出结果
fprintf('Stat (7) 均值：%.4f\n', mean(stat_eq7));
fprintf('Stat (8) 均值：%.4f\n', mean(stat_eq8));
fprintf('Stat (7) Bootstrap区间：[%.3f, %.3f]\n', ci7_low, ci7_high);
fprintf('Stat (8) Bootstrap区间：[%.3f, %.3f]\n', ci8_low, ci8_high);
fprintf('Stat (7) Power：%.3f\n', power_eq7);
fprintf('Stat (8) Power：%.3f\n', power_eq8);
