% Modello globale Self-Attention (Transformer-like) 
% per la predizione di cicli di isteresi.
% L'inferenza test è closed-loop (predizione 1-step iterativa).

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
fprintf('Generazione dataset globale...\n');
[train_curves, test_curves] = hysteresis_training();

%% ============================================================
%  2. SEQUENCE PREPARATION
% ============================================================
fprintf('\n=== PREPARAZIONE DATI SELFATTN-LSTM ===\n');

window_size = 70;      
sequence_stride = 20;  
val_split   = 0.2;     

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

rng(42, 'twister');
n_total = numel(X_all_cell);
perm_idx = randperm(n_total);
n_val = floor(val_split * n_total);
val_idx = perm_idx(1:n_val);
train_idx = perm_idx(n_val + 1:end);

X_train_cell = X_all_cell(train_idx);
Y_train_mat  = Y_all_mat(train_idx, :);
X_val_cell   = X_all_cell(val_idx);
Y_val_mat    = Y_all_mat(val_idx, :);

%% ============================================================
%  3. SELFATTN-LSTM ARCHITECTURE & TRAINING
% ============================================================

input_size = 3; 

% Multi-head self-attention on input features followed by LSTM reducer
layers = [
    sequenceInputLayer(input_size, 'Name', 'input')
    selfAttentionLayer(2, 32, 'Name', 'attention1') % 2 heads, 32 keys
    fullyConnectedLayer(64, 'Name', 'fc_hid')
    reluLayer('Name', 'relu')
    % flatten over sequence for final prediction
    lstmLayer(32, 'OutputMode','last','Name','seq_reducer') 
    fullyConnectedLayer(1, 'Name', 'output')
    regressionLayer('Name', 'regression')
];

options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 0.001, ...
    'ValidationData', {X_val_cell, Y_val_mat}, ...
    'ValidationFrequency', 200, ...
    'Plots', 'training-progress', ...
    'Verbose', true);

fprintf('\n=== ADDESTRAMENTO MODELLO SELFATTN-LSTM ===\n');
net = trainNetwork(X_train_cell, Y_train_mat, layers, options);

%% ============================================================
%  4. CLOSED-LOOP AUTOREGRESSIVE INFERENCE
% ============================================================
fprintf('\n=== GLOBAL TEST EVALUATION (SELFATTN-LSTM, CLOSED LOOP) ===\n');
fprintf('%-5s %-13s %-10s %-10s %-8s %-12s %-12s\n', ...
    'ID', 'Type', 'RMSE', 'MAE', 'R^2', 'Area Err %', 'CoerciveErr');

results = struct('test_id', {}, 'type_name', {}, 'mse', {}, 'rmse', {}, ...
    'mae', {}, 'r2', {}, 'area_err_pct', {}, 'coercive_err', {});

for test_id = 1:numel(test_curves)
    u_c = test_curves(test_id).u;
    du_dt_c = test_curves(test_id).du_dt;
    M_c = test_curves(test_id).M;
    t_c = test_curves(test_id).t;

    M_nn_c = zeros(size(M_c));
    M_nn_c(1:window_size) = M_c(1:window_size);

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
    area_pred     = abs(trapz(u_c, M_nn_c));
    if area_preisach > 0
        area_err_pct = abs(area_pred - area_preisach) / area_preisach * 100;
    else
        area_err_pct = NaN;
    end

    % CoerciveErr = mean(|Hc-_pred-Hc-_true|, |Hc+_pred-Hc+_true|)
    [hc_neg_true, hc_pos_true] = estimate_coercive_fields(u_c, M_c);
    [hc_neg_pred, hc_pos_pred] = estimate_coercive_fields(u_c, M_nn_c);
    coercive_err_c = mean(abs([hc_neg_pred - hc_neg_true, hc_pos_pred - hc_pos_true]), 'omitnan');
    
    test_curves(test_id).M_nn = M_nn_c;
    results(test_id).test_id = test_id;
    results(test_id).type_name = test_curves(test_id).name;
    results(test_id).mse = mse_c;
    results(test_id).rmse = rmse_c;
    results(test_id).mae = mae_c;
    results(test_id).r2 = r2_c;
    results(test_id).area_err_pct = area_err_pct;
    results(test_id).coercive_err = coercive_err_c;

    fprintf('%-5d %-13s %-10.4e %-10.4e %-8.4f %-12.2f %-12.4e\n', ...
        test_id, test_curves(test_id).name, rmse_c, mae_c, r2_c, area_err_pct, coercive_err_c);
end

fprintf('\nMetric formulas used:\n');
fprintf('R^2 = 1 - sum((y-yhat)^2)/sum((y-mean(y))^2)\n');
fprintf('MAE = mean(abs(y-yhat))\n');
fprintf('RMSE = sqrt(mean((y-yhat)^2))\n');
fprintf('Area Err %% = 100*abs(|int y du| - |int yhat du|)/|int y du|\n');
fprintf('CoerciveErr = mean(|Hc-_pred-Hc-_true|, |Hc+_pred-Hc+_true|)\n');

timestamp = datestr(now, 'yyyymmdd_HHMM');
save_folder = fullfile(repo_root, 'results_comparison');
if ~exist(save_folder, 'dir'), mkdir(save_folder); end
save(fullfile(save_folder, ['results_SelfAttn_LSTM_' timestamp '.mat']), 'results', 'test_curves');

unique_types = unique({results.type_name});
fprintf('\n=== RIEPILOGO PER TIPO DI ECCITAZIONE (SELFATTN-LSTM) ===\n');
for i = 1:numel(unique_types)
    type_name = unique_types{i};
    idx_type = find(strcmp({results.type_name}, type_name));
    
    mean_r2 = mean([results(idx_type).r2], 'omitnan');
    mean_area_err = mean([results(idx_type).area_err_pct], 'omitnan');
    fprintf('%-13s | Mean R^2 = %.4f | Mean Area Error = %.2f%%\n', ...
        type_name, mean_r2, mean_area_err);
end

fprintf('\n--- In-Domain Average (Types 1, 5, 6, 7) ---\n');
in_domain_types = {'Multi-Sine', 'Concentric Loops', 'FORC', 'Dense Minor Loops'};
idx_in = find(ismember({results.type_name}, in_domain_types));
fprintf('Mean R^2 = %.4f | Mean Area Error = %.2f%%\n', ...
        mean([results(idx_in).r2], 'omitnan'), mean([results(idx_in).area_err_pct], 'omitnan'));
        
fprintf('\n--- Extended Average (All Types) ---\n');
fprintf('Mean R^2 = %.4f | Mean Area Error = %.2f%%\n', ...
        mean([results.r2], 'omitnan'), mean([results.area_err_pct], 'omitnan'));

txt_filename = fullfile(save_folder, ['summary_SelfAttn_LSTM_' timestamp '.txt']);
fid = fopen(txt_filename, 'w');
if fid ~= -1
    fprintf(fid, '=== SELFATTN-LSTM PERFORMANCE SUMMARY (%s) ===\n\n', datestr(now));
    fprintf(fid, '%-5s %-13s %-10s %-10s %-8s %-12s %-12s\n', ...
        'ID', 'Type', 'RMSE', 'MAE', 'R^2', 'Area Err %', 'CoerciveErr');
    for test_id = 1:numel(results)
        fprintf(fid, '%-5d %-13s %-10.4e %-10.4e %-8.4f %-12.2f %-12.4e\n', ...
            test_id, results(test_id).type_name, results(test_id).rmse, ...
            results(test_id).mae, results(test_id).r2, ...
            results(test_id).area_err_pct, results(test_id).coercive_err);
    end
    fprintf(fid, '\n--- In-Domain Average (Types 1, 5, 6, 7) ---\n');
    fprintf(fid, 'Mean R^2 = %.4f | Mean Area Error = %.2f%%\n', ...
            mean([results(idx_in).r2], 'omitnan'), mean([results(idx_in).area_err_pct], 'omitnan'));
    fprintf(fid, '\n--- Extended Average (All Types) ---\n');
    fprintf(fid, 'Mean R^2 = %.4f | Mean Area Error = %.2f%%\n', ...
            mean([results.r2], 'omitnan'), mean([results.area_err_pct], 'omitnan'));
    fclose(fid);
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
