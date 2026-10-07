% Script to extract analytical metrics and generate LaTeX tables
% Formatted in formal booktabs style, ready for journal publication.

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

models_info = {
    'LSTM',            {'results_LSTM_*.mat'},                                         'results',     'M_nn';
    'SelfAttn-LSTM',   {'results_SelfAttn_LSTM_*.mat', 'results_SelfAttention_*.mat'}, 'results',     'M_nn';
    'Enc-SelfAttn',    {'results_Enc-SelfAttn_*.mat', 'results_Bahdanau_*.mat'},       'results',     'M_nn';
    'TFT',             {'results_TFT_2*.mat'},                                         'results_tft', 'M_pred_q50';
    'TFT (No Future)', {'results_TFT_NO_FUTURE_*.mat'},                                'results_tft', 'M_pred_q50';
};

num_models = size(models_info, 1);
model_data = cell(num_models, 1);
test_curves_data = cell(num_models, 1);
valid_models = false(num_models, 1);

%% 2. Load Data
fprintf('=== LOADING BENCHMARK DATA FOR LATEX TABLES ===\n');
for m = 1:num_models
    patterns = models_info{m, 2};
    files = [];
    for p = 1:numel(patterns)
        found = dir(fullfile(results_folder, patterns{p}));
        if ~isempty(found), files = [files; found]; end
    end
    if isempty(files), continue; end
    
    [~, sort_idx] = sort([files.datenum], 'descend');
    latest_file = fullfile(results_folder, files(sort_idx(1)).name);
    
    try
        data = load(latest_file);
        struct_name = models_info{m, 3};
        if isfield(data, struct_name)
            model_data{m} = data.(struct_name);
            valid_models(m) = true;
        elseif isfield(data, 'results')
            model_data{m} = data.results;
            valid_models(m) = true;
        end
        if isfield(data, 'test_curves')
            test_curves_data{m} = data.test_curves;
        end
        fprintf('[v] %-18s loaded\n', models_info{m, 1});
    catch
        fprintf('[x] Error loading %s\n', models_info{m, 1});
    end
end

if sum(valid_models) == 0
    error('No model results loaded.');
end

%% 3. Generate LaTeX Tables
ref_idx = find(valid_models, 1);
unique_types = unique({model_data{ref_idx}.type_name});

in_domain_types = {'Multi-Sine', 'Concentric Loops', 'FORC', 'Dense Minor Loops'};
out_domain_types = {'Stepwise', 'Damped Sine', 'Triangular'};

output_file = fullfile(results_folder, 'LaTeX_Tables.txt');
fid = fopen(output_file, 'w');

fprintf(fid, '%% ==========================================================\n');
fprintf(fid, '%% LATEX TABLES: REPRODUCED BENCHMARK RESULTS\n');
fprintf(fid, '%% Metrics: RMSE, MAE, R^2, Area Err (%%), Coercive Err, Remanent Err\n');
fprintf(fid, '%% ==========================================================\n\n');

write_latex_table(fid, 'Average Performance on In-Domain Excitations', 'tab:perf_indomain', in_domain_types, model_data, test_curves_data, valid_models, models_info);
write_latex_table(fid, 'Average Performance on Out-of-Domain Excitations', 'tab:perf_outdomain', out_domain_types, model_data, test_curves_data, valid_models, models_info);

for i = 1:numel(unique_types)
    t_name = unique_types{i};
    label_name = lower(strrep(strrep(t_name, ' ', '_'), '-', '_'));
    write_latex_table(fid, sprintf('Performance Comparison for %s Excitation', t_name), ['tab:perf_' label_name], {t_name}, model_data, test_curves_data, valid_models, models_info);
end

fclose(fid);
fprintf('\n=== LATEX TABLES EXPORTED SUCCESSFULLY ===\n');
fprintf('Saved to: %s\n\n', output_file);
type(output_file);

%% Helper Function
function write_latex_table(fid, caption_title, label, target_types, model_data, test_curves_data, valid_models, models_info)
    fprintf(fid, '\\begin{table}[htbp]\n');
    fprintf(fid, '\\centering\n');
    fprintf(fid, '\\caption{%s}\n', caption_title);
    fprintf(fid, '\\label{%s}\n', label);
    fprintf(fid, '\\begin{tabular}{@{}l c c c c c c@{}}\n');
    fprintf(fid, '\\toprule\n');
    fprintf(fid, '\\textbf{Model} & \\textbf{RMSE} & \\textbf{MAE} & \\textbf{$R^2$} & \\textbf{Area Err (\\%%)} & \\textbf{Coercive Err} & \\textbf{Remanent Err} \\n');
    fprintf(fid, '\\midrule\n');
    
    num_models = numel(valid_models);
    for m = 1:num_models
        if ~valid_models(m), continue; end
        
        idx = find(ismember({model_data{m}.type_name}, target_types));
        if isempty(idx), continue; end
        
        m_rmse = mean([model_data{m}(idx).rmse], 'omitnan');
        m_mae  = mean([model_data{m}(idx).mae], 'omitnan');
        m_r2   = mean([model_data{m}(idx).r2], 'omitnan');
        m_area = mean([model_data{m}(idx).area_err_pct], 'omitnan');
        m_coer = mean([model_data{m}(idx).coercive_err], 'omitnan');
        
        % Compute Remanent Error
        m_rem = NaN;
        tc = test_curves_data{m};
        if ~isempty(tc)
            pred_field = models_info{m, 4};
            rem_vals = [];
            for k = idx(:)'
                if k <= numel(tc) && isfield(tc(k), pred_field)
                    u = tc(k).u;
                    M_true = tc(k).M;
                    M_pred = tc(k).(pred_field);
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
                        rem_vals(end+1) = mean(c_errs);
                    end
                end
            end
            if ~isempty(rem_vals)
                m_rem = mean(rem_vals, 'omitnan');
            end
        end
        
        fprintf(fid, '%-18s & %.4f & %.4f & %.4f & %.2f & %.4f & %.4f \\n', ...
            models_info{m, 1}, m_rmse, m_mae, m_r2, m_area, m_coer, m_rem);
    end
    
    fprintf(fid, '\\bottomrule\n');
    fprintf(fid, '\\end{tabular}\n');
    fprintf(fid, '\\end{table}\n\n');
end
