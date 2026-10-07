% Detailed Single-Model Comparison vs Preisach Ground Truth
% Generates publication-ready M-H hysteresis loop plots for each waveform.

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
loaded_data = cell(num_models, 1);
valid_models = false(num_models, 1);

fprintf('\n=== LOADING DETAILED RESULTS ===\n');
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
        loaded_data{m} = load(latest_file);
        valid_models(m) = true;
        fprintf('[v] %-18s loaded from %s\n', models_info{m, 1}, files(sort_idx(1)).name);
    catch
        fprintf('[x] Error loading %s\n', latest_file);
    end
end

if sum(valid_models) == 0
    error('No valid models found in %s', results_folder);
end

fig_folder = fullfile(results_folder, 'figures_detailed');
if ~exist(fig_folder, 'dir'), mkdir(fig_folder); end

%% 2. Generate M-H Plots
in_domain_types = {'Concentric Loops', 'Dense Minor Loops', 'FORC', 'Multi-Sine'};

fprintf('\nGenerating publication-ready figures...\n');
for i = 1:numel(in_domain_types)
    type_name = in_domain_types{i};
    
    for m = 1:num_models
        if ~valid_models(m) || ~isfield(loaded_data{m}, 'test_curves')
            continue;
        end
        
        model_name = models_info{m, 1};
        pred_field = models_info{m, 4};
        cur_names = {loaded_data{m}.test_curves.name};
        idx = find(strcmp(cur_names, type_name));
        
        if isempty(idx), continue; end
        c_idx = idx(1);
        
        u_true = loaded_data{m}.test_curves(c_idx).u;
        M_true = loaded_data{m}.test_curves(c_idx).M;
        
        if ~isfield(loaded_data{m}.test_curves(c_idx), pred_field), continue; end
        M_pred = loaded_data{m}.test_curves(c_idx).(pred_field);
        
        fig = figure('Position', [150, 150, 650, 520], 'Visible', 'off');
        plot(u_true, M_true, 'k-', 'LineWidth', 2.0, 'DisplayName', 'Preisach Ground Truth'); hold on;
        plot(u_true, M_pred, 'r--', 'LineWidth', 1.8, 'DisplayName', model_name);
        
        xlabel('Magnetic Field u(t) [-]', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('Magnetization M(t) [-]', 'FontSize', 12, 'FontWeight', 'bold');
        title(sprintf('%s: %s vs Preisach', type_name, model_name), 'FontSize', 13, 'FontWeight', 'bold');
        legend('Location', 'NorthWest', 'FontSize', 11);
        grid on;
        xlim([-1.05, 1.05]);
        ylim([-1.05, 1.05]);
        
        safe_type = strrep(type_name, ' ', '_');
        safe_model = strrep(strrep(model_name, ' ', '_'), '(', '');
        safe_model = strrep(safe_model, ')', '');
        out_name = fullfile(fig_folder, sprintf('%s_vs_Preisach_%s.png', safe_type, safe_model));
        exportgraphics(fig, out_name, 'Resolution', 300);
        close(fig);
    end
end

fprintf('Detailed figures generated in: %s\n', fig_folder);
