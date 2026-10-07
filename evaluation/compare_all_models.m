% Comprehensive Model Benchmark for Magnetic Hysteresis Prediction
% Compares LSTM, SelfAttn-LSTM, Enc-SelfAttn, TFT, and TFT (No Future)
% across 7 excitation waveforms (4 In-Domain, 3 Out-of-Domain).

clear; close all; clc;

%% 1. Path Configuration
this_dir = fileparts(mfilename('fullpath'));
if isempty(this_dir), this_dir = pwd; end
repo_root = fullfile(this_dir, '..');

if exist(fullfile(repo_root, 'results_comparison'), 'dir')
    results_folder = fullfile(repo_root, 'results_comparison');
elseif exist('results_comparison', 'dir')
    results_folder = 'results_comparison';
else
    error('Directory results_comparison not found.');
end

% Model configurations aligned with paper terminology
models_info = {
    'LSTM',            {'results_LSTM_*.mat'},                                         'results',     'M_nn';
    'SelfAttn-LSTM',   {'results_SelfAttn_LSTM_*.mat', 'results_SelfAttention_*.mat'}, 'results',     'M_nn';
    'Enc-SelfAttn',    {'results_Enc-SelfAttn_*.mat', 'results_Bahdanau_*.mat'},       'results',     'M_nn';
    'TFT',             {'results_TFT_2*.mat'},                                         'results_tft', 'M_pred_q50';
    'TFT (No Future)', {'results_TFT_NO_FUTURE_*.mat'},                                'results_tft', 'M_pred_q50';
};

num_models = size(models_info, 1);
loaded_data = cell(num_models, 1);
valid_models = false(num_models, 1);

fprintf('\n=== LOADING BENCHMARK RESULTS ===\n');
for m = 1:num_models
    model_name = models_info{m, 1};
    patterns = models_info{m, 2};
    
    files = [];
    for p = 1:numel(patterns)
        found = dir(fullfile(results_folder, patterns{p}));
        if ~isempty(found)
            files = [files; found];
        end
    end
    
    if isempty(files)
        fprintf('[!] Results for %s not found.\n', model_name);
        continue;
    end
    
    [~, sort_idx] = sort([files.datenum], 'descend');
    latest_file = fullfile(results_folder, files(sort_idx(1)).name);
    
    try
        loaded_data{m} = load(latest_file);
        valid_models(m) = true;
        fprintf('[v] %-18s loaded from %s\n', model_name, files(sort_idx(1)).name);
    catch
        fprintf('[x] Error loading %s\n', latest_file);
    end
end

if sum(valid_models) == 0
    error('No valid model results found in %s', results_folder);
end

fig_folder = fullfile(results_folder, 'figures');
if ~exist(fig_folder, 'dir'), mkdir(fig_folder); end

%% 2. Metric Definitions
fprintf('\n=== EVALUATION METRICS ===\n');
fprintf('R^2 = 1 - sum((y - yhat)^2) / sum((y - mean(y))^2)\n');
fprintf('MAE = mean(abs(y - yhat))\n');
fprintf('RMSE = sqrt(mean((y - yhat)^2))\n');
fprintf('Area Error %% = 100 * abs(|int y du| - |int yhat du|) / |int y du|\n');
fprintf('Coercive Error = mean(|Hc-_pred - Hc-_true|, |Hc+_pred - Hc+_true|)\n');
fprintf('Remanent Error = mean(|M_pred - M_true|) evaluated at u = 0 zero-crossings.\n');

metrics_info = {
    'r2',            'R^2',                      true;
    'mae',           'MAE',                      false;
    'rmse',          'RMSE',                     false;
    'area_err_pct',  'Area Error [%]',           false;
    'coercive_err',  'Coercive Error',           false;
    'remanent_err',  'Remanent Magnetization Err', false;
};

unique_types = {};
for m = 1:num_models
    if valid_models(m)
        res = loaded_data{m}.(models_info{m, 3});
        if isfield(loaded_data{m}, 'test_curves')
            c_names = {loaded_data{m}.test_curves.name};
            unique_types = [unique_types, c_names];
        elseif isfield(res, 'type_name')
            unique_types = [unique_types, {res.type_name}];
        end
    end
end
unique_types = unique(unique_types);

in_domain_types = {'Multi-Sine', 'Concentric Loops', 'FORC', 'Dense Minor Loops'};
idx_in_domain = find(ismember(unique_types, in_domain_types));
idx_extended = 1:numel(unique_types);

num_metrics = size(metrics_info, 1);
metric_matrices = cell(num_metrics, 1);
for q = 1:num_metrics
    metric_matrices{q} = NaN(numel(unique_types), num_models);
end

for q = 1:num_metrics
    metric_field = metrics_info{q, 1};
    metric_label = metrics_info{q, 2};

    for i = 1:numel(unique_types)
        type_name = unique_types{i};
        for m = 1:num_models
            if valid_models(m)
                res = loaded_data{m}.(models_info{m, 3});
                tc = [];
                if isfield(loaded_data{m}, 'test_curves')
                    tc = loaded_data{m}.test_curves;
                    idx_type = find(strcmp({tc.name}, type_name));
                elseif isfield(res, 'type_name')
                    idx_type = find(strcmp({res.type_name}, type_name));
                else
                    idx_type = [];
                end
                mean_metric = extract_mean_metric(res, idx_type, metric_field, tc, models_info{m, 4});
                metric_matrices{q}(i, m) = mean_metric;
            end
        end
    end

    % ---- Print In-Domain Test Table ----
    fprintf('\n=== [In-Domain Test] %s ===\n', metric_label);
    header = sprintf('%-18s |', 'Waveform Type');
    for m = 1:num_models
        if valid_models(m)
            header = [header, sprintf(' %-16s |', models_info{m, 1})];
        end
    end
    fprintf('%s\n', header);
    fprintf('%s\n', repmat('-', 1, length(header)));

    for i = idx_in_domain(:)'
        row_str = sprintf('%-18s |', unique_types{i});
        for m = 1:num_models
            if valid_models(m)
                row_str = [row_str, sprintf(' %-16.4f |', metric_matrices{q}(i, m))];
            end
        end
        fprintf('%s\n', row_str);
    end
    fprintf('%s\n', repmat('-', 1, length(header)));
    
    row_str = sprintf('%-18s |', 'IN-DOMAIN MEAN');
    for m = 1:num_models
        if valid_models(m)
            row_str = [row_str, sprintf(' %-16.4f |', mean(metric_matrices{q}(idx_in_domain, m), 'omitnan'))];
        end
    end
    fprintf('%s\n', row_str);

    % ---- Print Extended Test Table ----
    fprintf('\n=== [Extended Test (All 7 Waveforms)] %s ===\n', metric_label);
    fprintf('%s\n', header);
    fprintf('%s\n', repmat('-', 1, length(header)));
    for i = idx_extended(:)'
        row_str = sprintf('%-18s |', unique_types{i});
        for m = 1:num_models
            if valid_models(m)
                row_str = [row_str, sprintf(' %-16.4f |', metric_matrices{q}(i, m))];
            end
        end
        fprintf('%s\n', row_str);
    end
    fprintf('%s\n', repmat('-', 1, length(header)));
    
    row_str = sprintf('%-18s |', 'EXTENDED MEAN');
    for m = 1:num_models
        if valid_models(m)
            row_str = [row_str, sprintf(' %-16.4f |', mean(metric_matrices{q}(idx_extended, m), 'omitnan'))];
        end
    end
    fprintf('%s\n', row_str);
end

fprintf('\n=== Benchmark evaluation completed successfully ===\n');

%% Helper Function
function mean_metric = extract_mean_metric(res_struct, idx_type, field_name, test_curves, pred_field)
    mean_metric = NaN;
    if isempty(idx_type), return; end

    if strcmp(field_name, 'remanent_err')
        if ~isempty(test_curves)
            vals = [];
            for k = idx_type(:)'
                if k <= numel(test_curves) && isfield(test_curves(k), pred_field)
                    u = test_curves(k).u;
                    M_true = test_curves(k).M;
                    M_pred = test_curves(k).(pred_field);
                    idx_zc = find((u(1:end-1) .* u(2:end) <= 0) & (u(1:end-1) ~= u(2:end)));
                    if ~isempty(idx_zc)
                        c_errs = zeros(numel(idx_zc), 1);
                        for z = 1:numel(idx_zc)
                            iz = idx_zc(z);
                            u1 = u(iz); u2 = u(iz+1);
                            frac = -u1 / (u2 - u1);
                            m_true_0 = M_true(iz) + frac * (M_true(iz+1) - M_true(iz));
                            m_pred_0 = M_pred(iz) + frac * (M_pred(iz+1) - M_pred(iz));
                            c_errs(z) = abs(m_pred_0 - m_true_0);
                        end
                        vals(end+1) = mean(c_errs);
                    end
                end
            end
            if ~isempty(vals)
                mean_metric = mean(vals, 'omitnan');
                return;
            end
        end
    end

    if isempty(res_struct) || ~isfield(res_struct, field_name)
        return;
    end
    vals = [res_struct(idx_type).(field_name)];
    mean_metric = mean(vals, 'omitnan');
end
