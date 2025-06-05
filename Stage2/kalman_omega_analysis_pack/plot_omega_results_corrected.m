
%% plot_omega_results_corrected.m
% Load and visualize results from kalman_omega_analysis_corrected.m

load('omega_analysis_corrected_results.mat');  % Loads stat7_matrix, stat8_matrix, omega_list

%% Compute Mean Across Simulations
mean_stat7 = mean(stat7_matrix, 2);
mean_stat8 = mean(stat8_matrix, 2);

%% Display Summary Table
T = table(omega_list(:), mean_stat7, mean_stat8, ...
    'VariableNames', {'Omega', 'Mean_Stat7', 'Mean_Stat8'});
disp('=== Summary Table of stat(7) and stat(8) ===');
disp(T);

%% Line Plot of Means
figure;
plot(omega_list, mean_stat7, '-o', 'LineWidth', 2); hold on;
plot(omega_list, mean_stat8, '-s', 'LineWidth', 2);
xlabel('\omega (Signal Noise Std Dev)');
ylabel('Statistic Value');
title('Mean stat(7) and stat(8) vs. \omega');
legend('stat(7)', 'stat(8)', 'Location', 'best');
grid on;

%% Boxplot of stat(7)
figure;
boxplot(stat7_matrix', 'Labels', string(omega_list));
title('Distribution of stat(7) under Different \omega');
xlabel('\omega');
ylabel('stat(7)');
grid on;

%% Boxplot of stat(8)
figure;
boxplot(stat8_matrix', 'Labels', string(omega_list));
title('Distribution of stat(8) under Different \omega');
xlabel('\omega');
ylabel('stat(8)');
grid on;
