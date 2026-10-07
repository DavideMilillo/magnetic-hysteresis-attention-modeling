% Deep Recurrent LSTM Baseline for Magnetic Hysteresis Prediction
% Utilizza i dati generati centralmente dal modulo hysteresis_training.m
% Inferenza closed-loop senza teacher forcing in fase di test

clear; close all; clc;

% Add repository paths automatically
this_dir = fileparts(mfilename('fullpath'));
if isempty(this_dir), this_dir = pwd; end
repo_root = fullfile(this_dir, '..');
addpath(fullfile(repo_root, 'data'));
addpath(fullfile(repo_root, 'models'));
addpath(fullfile(repo_root, 'models', 'layers'));


%% ============================================================
%  1. DATASET ACQUISITION
% ============================================================
fprintf('Generating global dataset via hysteresis_training...\n');
[train_curves, test_curves] = hysteresis_training();

%% ============================================================
%  2. SEQUENCE PREPARATION PER LSTM
% ============================================================
fprintf('\n=== PREPARING LSTM TRAINING SEQUENCES ===\n');

window_size = 70;      % Finestra temporale (campioni)
sequence_stride = 20;  % Usa una finestra ogni 20 campioni per ridurre il dataset
val_split   = 0.2;     % Validation split globale %

X_all_cell    = {};
Y_all_mat     = [];

for i = 1:numel(train_curves)
    u_c = train_curves(i).u;
    du_dt_c = train_curves(i).du_dt;
    M_c = train_curves(i).M;

    [X_c, Y_c] = create_sequences(u_c, du_dt_c, M_c, window_size, sequence_stride);

    for seq_idx = 1:size(X_c, 1)
        X_all_cell{end+1, 1} = squeeze(X_c(seq_idx, :, :))';
    end
    Y_all_mat = [Y_all_mat; Y_c];
end

if any(isnan(Y_all_mat))
    error('Il dataset globale contiene NaN nelle target.');
end

rng(42, 'twister'); % Riproducibilità dello split
n_total = numel(X_all_cell);
perm_idx = randperm(n_total);
n_val = floor(val_split * n_total);
val_idx = perm_idx(1:n_val);
train_idx = perm_idx(n_val + 1:end);

X_train_cell = X_all_cell(train_idx);
Y_train_mat  = Y_all_mat(train_idx, :);
X_val_cell   = X_all_cell(val_idx);
Y_val_mat    = Y_all_mat(val_idx, :);

fprintf('Training sequences:   %d\n', numel(X_train_cell));
fprintf('Validation sequences: %d\n', numel(X_val_cell));
fprintf('Lookback window:      %d steps\n', window_size);

%% ============================================================
%  3. LSTM ARCHITECTURE & TRAINING
% ============================================================

input_size = 3; % u, du/dt, M_past
early_stopping_patience = 50; 

layers = [
    sequenceInputLayer(input_size, 'Name', 'input')
    lstmLayer(300, 'OutputMode', 'sequence', 'Name', 'lstm1')
    dropoutLayer(0.1, 'Name', 'dropout1')
    lstmLayer(100, 'OutputMode', 'last', 'Name', 'lstm2')
    dropoutLayer(0.1, 'Name', 'dropout2')
    fullyConnectedLayer(16, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(1, 'Name', 'output')
    regressionLayer('Name', 'regression')
];

options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 0.00005, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 30, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {X_val_cell, Y_val_mat}, ...
    'ValidationFrequency', 200, ...
    'ValidationPatience', early_stopping_patience, ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'ExecutionEnvironment', 'auto');

fprintf('\n=== ADDESTRAMENTO MODELLO GLOBALE ===\n');
net = trainNetwork(X_train_cell, Y_train_mat, layers, options);

%% ============================================================
%  4. CLOSED-LOOP AUTOREGRESSIVE INFERENCE SUL TEST SET FISSO
% ============================================================

fprintf('\n=== VALUTAZIONE TEST SET GLOBALE (CLOSED LOOP) ===\n');
fprintf('%-5s %-13s %-10s %-10s %-8s %-12s %-12s\n', ...
    'ID', 'Tipo', 'RMSE', 'MAE', 'R^2', 'Area Err %', 'CoerciveErr');

results = struct('test_id', {}, 'type_name', {}, 'mse', {}, 'rmse', {}, ...
    'mae', {}, 'r2', {}, 'area_err_pct', {}, 'coercive_err', {});

for test_id = 1:numel(test_curves)
    u_c = test_curves(test_id).u;
    du_dt_c = test_curves(test_id).du_dt;
    M_c = test_curves(test_id).M;
    t_c = test_curves(test_id).t;

    M_nn_c = zeros(size(M_c));
    M_nn_c(1:window_size) = M_c(1:window_size); % Bootstrap iniziale

    % Inferenza ricorsiva (closed loop)
    for k = window_size + 1:length(M_c)
        u_hist = u_c(k-window_size+1:k);
        du_hist = du_dt_c(k-window_size+1:k);
        M_hist = M_nn_c(k-window_size:k-1);
        X_step = [u_hist; du_hist; M_hist];

        M_pred_k = predict(net, X_step, 'ExecutionEnvironment', 'cpu');
        M_nn_c(k) = M_pred_k;
    end

    idx = window_size + 1:length(M_c);
    err = M_c(idx) - M_nn_c(idx);

    % RMSE = sqrt(mean((y - yhat).^2))
    mse_c  = mean(err.^2);
    rmse_c = sqrt(mse_c);
    % MAE = mean(abs(y - yhat))
    mae_c  = mean(abs(err));

    % R^2 = 1 - sum((y - yhat).^2) / sum((y - mean(y)).^2)
    denom_r2 = sum((M_c(idx) - mean(M_c(idx))).^2);
    if denom_r2 > 0
        r2_c = 1 - sum(err.^2) / denom_r2;
    else
        r2_c = NaN;
    end

    % Area error %% = 100 * abs(|int y du| - |int yhat du|) / |int y du|
    area_preisach = abs(trapz(u_c, M_c));
    area_lstm     = abs(trapz(u_c, M_nn_c));
    if area_preisach > 0
        area_err_pct = abs(area_lstm - area_preisach) / area_preisach * 100;
    else
        area_err_pct = NaN;
    end

    % Coercive field error from M(u)=0 crossings.
    % Steps:
    % 1) Find zero-crossing u values via linear interpolation on segments where M changes sign.
    % 2) Use Hc- = min(u_cross), Hc+ = max(u_cross).
    % 3) CoerciveErr = mean([abs(Hc-_pred - Hc-_true), abs(Hc+_pred - Hc+_true)]).
    [hc_neg_true, hc_pos_true] = estimate_coercive_fields(u_c, M_c);
    [hc_neg_pred, hc_pos_pred] = estimate_coercive_fields(u_c, M_nn_c);
    coercive_err_c = mean(abs([hc_neg_pred - hc_neg_true, hc_pos_pred - hc_pos_true]), 'omitnan');

    results(test_id).test_id        = test_id;
    results(test_id).type_name      = test_curves(test_id).name;
    results(test_id).mse            = mse_c;
    results(test_id).rmse           = rmse_c;
    results(test_id).mae            = mae_c;
    results(test_id).r2             = r2_c;
    results(test_id).area_err_pct   = area_err_pct;
    results(test_id).coercive_err   = coercive_err_c;

    test_curves(test_id).M_nn = M_nn_c;

    fprintf('%-5d %-13s %-10.4e %-10.4e %-8.4f %-12.2f %-12.4e\n', ...
        test_id, test_curves(test_id).name, rmse_c, mae_c, r2_c, area_err_pct, coercive_err_c);
end

%% ============================================================
%  5. VISUALIZZAZIONE RISULTATI
% ============================================================

unique_types = unique({test_curves.name});

for i = 1:numel(unique_types)
    type_name = unique_types{i};
    % Trova tutti gli indici di test con questo tipo
    idx_type = find(strcmp({test_curves.name}, type_name));

    figure('Position', [60, 80, 1450, 650], ...
        'Name', sprintf('Test Closed-Loop - %s', type_name));

    for local_plot_idx = 1:numel(idx_type)
        curve_idx = idx_type(local_plot_idx);
        u_c = test_curves(curve_idx).u;
        M_c = test_curves(curve_idx).M;
        t_c = test_curves(curve_idx).t;
        M_nn_c = test_curves(curve_idx).M_nn;

        subplot(2, numel(idx_type), local_plot_idx);
        plot(u_c, M_c, 'b-', 'LineWidth', 1.5); hold on;
        plot(u_c, M_nn_c, 'r--', 'LineWidth', 1.5);
        xlabel('u(t)', 'FontSize', 11);
        ylabel('M(t)', 'FontSize', 11);
        title(sprintf('%s Test %d\nR^2=%.4f | \\DeltaArea=%.2f%%', ...
            type_name, local_plot_idx, ...
            results(curve_idx).r2, results(curve_idx).area_err_pct), 'FontSize', 11);
        legend('Preisach', 'LSTM', 'Location', 'best', 'FontSize', 9);
        grid on;

        subplot(2, numel(idx_type), numel(idx_type) + local_plot_idx);
        plot(t_c, M_c, 'b-', 'LineWidth', 1.5); hold on;
        plot(t_c, M_nn_c, 'r--', 'LineWidth', 1.5);
        xlabel('Time [s]', 'FontSize', 11);
        ylabel('M(t)', 'FontSize', 11);
        title(sprintf('RMSE=%.4e | MAE=%.4e', ...
            results(curve_idx).rmse, results(curve_idx).mae), 'FontSize', 11);
        legend('Preisach', 'LSTM', 'Location', 'best', 'FontSize', 9);
        grid on;
    end
end

fprintf('\n=== RIEPILOGO PER TIPO DI ECCITAZIONE ===\n');
for i = 1:numel(unique_types)
    type_name = unique_types{i};
    idx_type = find(strcmp({results.type_name}, type_name));
    
    mean_r2 = mean([results(idx_type).r2], 'omitnan');
    mean_area_err = mean([results(idx_type).area_err_pct], 'omitnan');
    fprintf('%-13s | Mean R^2 = %.4f | Mean Area Error = %.2f%%\n', ...
        type_name, mean_r2, mean_area_err);
end

fprintf('\nMetric formulas used:\n');
fprintf('R^2 = 1 - sum((y-yhat)^2)/sum((y-mean(y))^2)\n');
fprintf('MAE = mean(abs(y-yhat))\n');
fprintf('RMSE = sqrt(mean((y-yhat)^2))\n');
fprintf('Area Err %% = 100*abs(|int y du| - |int yhat du|)/|int y du|\n');
fprintf('CoerciveErr = mean(|Hc-_pred-Hc-_true|, |Hc+_pred-Hc+_true|)\n');

%% ============================================================
%  6. SALVATAGGIO RISULTATI
% ============================================================
timestamp = datestr(now, 'yyyymmdd_HHMM');
save_folder = fullfile(repo_root, 'results_comparison');
if ~exist(save_folder, 'dir'), mkdir(save_folder); end

% 1. Salvataggio struct dei risultati e delle curve (.mat)
save(fullfile(save_folder, ['results_LSTM_' timestamp '.mat']), 'results', 'test_curves');

% 2. Esportazione summary testuale (.txt)
txt_filename = fullfile(save_folder, ['summary_LSTM_' timestamp '.txt']);
fid = fopen(txt_filename, 'w');
if fid ~= -1
    fprintf(fid, '=== LSTM PERFORMANCE SUMMARY (%s) ===\n\n', datestr(now));
    fprintf(fid, '%-5s %-13s %-10s %-10s %-8s %-12s %-12s\n', ...
        'ID', 'Type', 'RMSE', 'MAE', 'R^2', 'Area Err %', 'CoerciveErr');
    for test_id = 1:numel(results)
        fprintf(fid, '%-5d %-13s %-10.4e %-10.4e %-8.4f %-12.2f %-12.4e\n', ...
            test_id, results(test_id).type_name, results(test_id).rmse, ...
            results(test_id).mae, results(test_id).r2, ...
            results(test_id).area_err_pct, results(test_id).coercive_err);
    end
    fprintf(fid, '\nMetric formulas used:\n');
    fprintf(fid, 'R^2 = 1 - sum((y-yhat)^2)/sum((y-mean(y))^2)\n');
    fprintf(fid, 'MAE = mean(abs(y-yhat))\n');
    fprintf(fid, 'RMSE = sqrt(mean((y-yhat)^2))\n');
    fprintf(fid, 'Area Err %% = 100*abs(|int y du| - |int yhat du|)/|int y du|\n');
    fprintf(fid, 'CoerciveErr = mean(|Hc-_pred-Hc-_true|, |Hc+_pred-Hc+_true|)\n');
    fprintf(fid, '\n--- Medie per Tipo ---\n');
    for i = 1:numel(unique_types)
        type_name = unique_types{i};
        idx_type = find(strcmp({results.type_name}, type_name));
        fprintf(fid, '%-13s | Mean R^2 = %.4f | Mean Area Error = %.2f%%\n', ...
            type_name, mean([results(idx_type).r2], 'omitnan'), ...
            mean([results(idx_type).area_err_pct], 'omitnan'));
    end
    
    fprintf(fid, '\n--- In-Domain Average (Types 1, 5, 6, 7) ---\n');
    in_domain_types = {'Multi-Sine', 'Concentric Loops', 'FORC', 'Dense Minor Loops'};
    idx_in = find(ismember({results.type_name}, in_domain_types));
    fprintf(fid, 'Mean R^2 = %.4f | Mean Area Error = %.2f%%\n', ...
            mean([results(idx_in).r2], 'omitnan'), mean([results(idx_in).area_err_pct], 'omitnan'));
            
    fprintf(fid, '\n--- Extended Average (All Types) ---\n');
    fprintf(fid, 'Mean R^2 = %.4f | Mean Area Error = %.2f%%\n', ...
            mean([results.r2], 'omitnan'), mean([results.area_err_pct], 'omitnan'));

    fclose(fid);
    fprintf('\nRisultati salvati in: %s\n', save_folder);
end

%% ============================================================
%  LOCAL HELPER FUNCTIONS
% ============================================================
function [X, Y] = create_sequences(u, du_dt, M, window_size, sequence_stride)
    start_indices = 1:sequence_stride:(length(u) - window_size);
    n_samples = numel(start_indices);
    X = zeros(n_samples, window_size, 3);
    Y = zeros(n_samples, 1);

    for i = 1:n_samples
        start_idx = start_indices(i);
        k = start_idx + window_size;
        
        X(i, :, 1) = u(k-window_size+1:k);
        X(i, :, 2) = du_dt(k-window_size+1:k);
        X(i, :, 3) = M(k-window_size:k-1);
        Y(i) = M(k);
    end
end

function [hc_neg, hc_pos] = estimate_coercive_fields(u, M)
    idx = find((M(1:end-1) .* M(2:end) <= 0) & (M(1:end-1) ~= M(2:end)));
    if isempty(idx)
        hc_neg = NaN;
        hc_pos = NaN;
        return;
    end

    u_cross = zeros(numel(idx), 1);
    for i = 1:numel(idx)
        k = idx(i);
        m1 = M(k); m2 = M(k+1);
        u1 = u(k); u2 = u(k+1);
        u_cross(i) = u1 - m1 * (u2 - u1) / (m2 - m1);
    end

    hc_neg = min(u_cross);
    hc_pos = max(u_cross);
end
