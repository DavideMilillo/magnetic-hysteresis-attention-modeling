% Script standalone per visualizzare mappe di attenzione (importanza temporale)
% per i modelli attention-based senza modificare gli script originali.
% Nota: per mantenere la riproducibilita dei vecchi risultati, train_Enc_SelfAttn.m,
% train_SelfAttn_LSTM.m e train_TFT.m non vengono alterati.

clear; close all; clc;

%% ============================================================
%  1) CONFIGURAZIONE
% ============================================================
window_size = 70;
sequence_stride = 20;
val_split = 0.2;
max_epochs = 100;
mini_batch_size = 64;
% Number of future time steps predicted per TFT block.
tft_prediction_horizon = 25;
tft_type_levels = {'Multi-Sine', 'Damped Sine', 'Stepwise', 'Triangular', 'Concentric Loops', 'FORC', 'Dense Minor Loops'};
tft_quantiles = [0.1; 0.5; 0.9];
tft_num_hidden_units = 64;
tft_num_attention_heads = 4;
tft_num_categories = 7;
tft_num_output_quantiles = 3;
tft_dropout_probability = 0.1;

this_dir = fileparts(mfilename('fullpath'));
if isempty(this_dir), this_dir = pwd; end
repo_root = fullfile(this_dir, '..');
addpath(fullfile(repo_root, 'data'));
addpath(fullfile(repo_root, 'models'));
addpath(fullfile(repo_root, 'models', 'layers'));
results_folder = fullfile(repo_root, 'results_comparison');
fig_folder = fullfile(results_folder, 'figures', 'attention_maps');
if ~exist(fig_folder, 'dir')
    mkdir(fig_folder);
end

rng(42, 'twister');

%% ============================================================
%  2) DATASET GLOBALE
% ============================================================
fprintf('Generazione dataset globale tramite hysteresis_training...\n');
[train_curves, test_curves] = hysteresis_training();

fprintf('Preparazione sequenze train/validation...\n');
[X_all_cell, Y_all_mat] = build_all_sequences(train_curves, window_size, sequence_stride);

n_total = numel(X_all_cell);
perm_idx = randperm(n_total);
n_val = floor(val_split * n_total);
val_idx = perm_idx(1:n_val);
train_idx = perm_idx(n_val + 1:end);

X_train_cell = X_all_cell(train_idx);
Y_train_mat = Y_all_mat(train_idx, :);
X_val_cell = X_all_cell(val_idx);
Y_val_mat = Y_all_mat(val_idx, :);

options = trainingOptions('adam', ...
    'MaxEpochs', max_epochs, ...
    'MiniBatchSize', mini_batch_size, ...
    'InitialLearnRate', 0.001, ...
    'ValidationData', {X_val_cell, Y_val_mat}, ...
    'ValidationFrequency', 200, ...
    'Plots', 'training-progress', ...
    'Verbose', true);

%% ============================================================
%  3) MODELLI ATTENTION-BASED DA ANALIZZARE
% ============================================================
models = {
    'SelfAttn-LSTM', @build_self_attention_model;
    'Enc-SelfAttn',      @build_bahdanau_model;
};

curve_id = choose_curve_with_most_reversals(test_curves);
fprintf('Curva selezionata per interpretabilita: ID %d (%s)\n', curve_id, test_curves(curve_id).name);

for m = 1:size(models, 1)
    model_name = models{m, 1};
    model_builder = models{m, 2};

    fprintf('\n=== TRAINING %s ===\n', model_name);
    net = trainNetwork(X_train_cell, Y_train_mat, model_builder(), options);

    fprintf('Inferenza closed-loop + stima attenzione (%s)...\n', model_name);
    [M_pred, attention_series, attention_map] = closed_loop_with_occlusion_attention(net, test_curves(curve_id), window_size);

    u = test_curves(curve_id).u;
    t = test_curves(curve_id).t;
    M_true = test_curves(curve_id).M;
    du_dt = test_curves(curve_id).du_dt;

    reversal_idx = find_reversal_points(du_dt);
    segment_idx = choose_segment_around_reversals(length(t), reversal_idx, window_size);

    fig = figure('Position', [120, 80, 1200, 850], 'Name', ['Attention map - ', model_name]);
    tl = tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile(tl, 1);
    h_true_loop = plot(u(segment_idx), M_true(segment_idx), 'k-', 'LineWidth', 2.0); hold on;
    h_pred_loop = plot(u(segment_idx), M_pred(segment_idx), 'r--', 'LineWidth', 1.2);
    h_rev_loop = [];
    rev_seg = reversal_idx(reversal_idx >= segment_idx(1) & reversal_idx <= segment_idx(end));
    if ~isempty(rev_seg)
        h_rev_loop = scatter(u(rev_seg), M_true(rev_seg), 45, 'b', 'filled', 'DisplayName', 'Reversal points');
    end
    grid on;
    xlabel('u(t) Applied Field'); ylabel('M(t)');
    title(sprintf('%s - Hysteresis segment M(u)', model_name));
    add_legend_with_optional_reversals(h_true_loop, h_pred_loop, h_rev_loop);

    nexttile(tl, 2);
    h_true_t = plot(t, M_true, 'k-', 'LineWidth', 1.3); hold on;
    h_pred_t = plot(t, M_pred, 'r--', 'LineWidth', 1.1);
    h_rev_t = draw_reversal_lines(t, reversal_idx, ':b');
    grid on;
    xlabel('Time t [s]'); ylabel('M(t)');
    title('Temporal evolution of magnetization');
    add_legend_with_optional_reversals(h_true_t, h_pred_t, h_rev_t);

    nexttile(tl, 3);
    h_att = plot(t, attention_series, 'm-', 'LineWidth', 1.4); hold on;
    h_rev_att = draw_reversal_lines(t, reversal_idx, ':b');
    grid on;
    xlabel('Time t [s]'); ylabel('Attention weight');
    title('Temporal attention weights (occlusion-based estimate)');
    if isempty(h_rev_att)
        legend(h_att, {'Attention weight'}, 'Location', 'best');
    else
        legend([h_att, h_rev_att], {'Attention weight', 'Reversal points'}, 'Location', 'best');
    end

    out_png = fullfile(fig_folder, sprintf('AttentionMap_%s_curve%d.png', model_name, curve_id));
    exportgraphics(fig, out_png, 'Resolution', 300);

    out_mat = fullfile(fig_folder, sprintf('AttentionWeights_%s_curve%d.mat', model_name, curve_id));
    save(out_mat, 'model_name', 'curve_id', 'M_true', 'M_pred', 'u', 't', 'du_dt', ...
        'attention_series', 'attention_map', 'reversal_idx', 'segment_idx');

    fprintf('[OK] Salvati: %s\n', out_png);
    fprintf('[OK] Salvati: %s\n', out_mat);
end

%% ============================================================
%  4) TFT (ATTENTION-BASED) - BEST EFFORT
% ============================================================
if exist('createTFTNetwork', 'file') == 2 && exist('trainnet', 'file') == 2
    fprintf('\n=== TRAINING TFT ===\n');
    [trainedNetTFT, tipo_cat_levels] = train_tft_interpretability_model( ...
        train_curves, window_size, tft_prediction_horizon, sequence_stride, val_split, mini_batch_size, max_epochs, tft_type_levels, ...
        tft_quantiles, tft_num_hidden_units, tft_num_attention_heads, tft_num_categories, tft_num_output_quantiles, tft_dropout_probability);

    if ~isempty(trainedNetTFT)
        fprintf('Inferenza closed-loop + stima attenzione (TFT)...\n');
        [M_pred_tft, attention_series_tft, attention_map_tft] = closed_loop_tft_with_occlusion_attention( ...
            trainedNetTFT, test_curves(curve_id), window_size, tft_prediction_horizon, tipo_cat_levels);

        u = test_curves(curve_id).u;
        t = test_curves(curve_id).t;
        M_true = test_curves(curve_id).M;
        du_dt = test_curves(curve_id).du_dt;

        reversal_idx = find_reversal_points(du_dt);
        segment_idx = choose_segment_around_reversals(length(t), reversal_idx, window_size);

        fig = figure('Position', [120, 80, 1200, 850], 'Name', 'Attention map - TFT');
        tl = tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

        nexttile(tl, 1);
        h_true_loop = plot(u(segment_idx), M_true(segment_idx), 'k-', 'LineWidth', 2.0); hold on;
        h_pred_loop = plot(u(segment_idx), M_pred_tft(segment_idx), 'r--', 'LineWidth', 1.2);
        h_rev_loop = [];
        rev_seg = reversal_idx(reversal_idx >= segment_idx(1) & reversal_idx <= segment_idx(end));
        if ~isempty(rev_seg)
            h_rev_loop = scatter(u(rev_seg), M_true(rev_seg), 45, 'b', 'filled', 'DisplayName', 'Reversal points');
        end
        grid on;
        xlabel('u(t) Applied Field'); ylabel('M(t)');
        title('TFT - Hysteresis segment M(u)');
        add_legend_with_optional_reversals(h_true_loop, h_pred_loop, h_rev_loop);

        nexttile(tl, 2);
        h_true_t = plot(t, M_true, 'k-', 'LineWidth', 1.3); hold on;
        h_pred_t = plot(t, M_pred_tft, 'r--', 'LineWidth', 1.1);
        h_rev_t = draw_reversal_lines(t, reversal_idx, ':b');
        grid on;
        xlabel('Time t [s]'); ylabel('M(t)');
        title('Temporal evolution of magnetization');
        add_legend_with_optional_reversals(h_true_t, h_pred_t, h_rev_t);

        nexttile(tl, 3);
        h_att = plot(t, attention_series_tft, 'm-', 'LineWidth', 1.4); hold on;
        h_rev_att = draw_reversal_lines(t, reversal_idx, ':b');
        grid on;
        xlabel('Time t [s]'); ylabel('Attention weight');
        title('Temporal attention weights (occlusion-based estimate)');
        if isempty(h_rev_att)
            legend(h_att, {'Attention weight'}, 'Location', 'best');
        else
            legend([h_att, h_rev_att], {'Attention weight', 'Reversal points'}, 'Location', 'best');
        end

        out_png = fullfile(fig_folder, sprintf('AttentionMap_%s_curve%d.png', 'TFT', curve_id));
        exportgraphics(fig, out_png, 'Resolution', 300);

        out_mat = fullfile(fig_folder, sprintf('AttentionWeights_%s_curve%d.mat', 'TFT', curve_id));
        save(out_mat, 'curve_id', 'M_true', 'M_pred_tft', 'u', 't', 'du_dt', ...
            'attention_series_tft', 'attention_map_tft', 'reversal_idx', 'segment_idx');

        fprintf('[OK] Salvati: %s\n', out_png);
        fprintf('[OK] Salvati: %s\n', out_mat);
    else
        fprintf('[WARN] TFT non addestrato. Sezione attention map TFT saltata.\n');
    end
else
    fprintf('[WARN] createTFTNetwork/trainnet non disponibili. Sezione attention map TFT saltata.\n');
end

fprintf('\nCompletato. Figure in: %s\n', fig_folder);

%% ============================================================
%  FUNZIONI LOCALI
% ============================================================
function layers = build_self_attention_model()
input_size = 3;
layers = [
    sequenceInputLayer(input_size, 'Name', 'input')
    selfAttentionLayer(2, 32, 'Name', 'attention1')
    fullyConnectedLayer(64, 'Name', 'fc_hid')
    reluLayer('Name', 'relu')
    lstmLayer(32, 'OutputMode', 'last', 'Name', 'seq_reducer')
    fullyConnectedLayer(1, 'Name', 'output')
    regressionLayer('Name', 'regression')
];
end

function lgraph = build_bahdanau_model()
input_size = 3;
lgraph = layerGraph();
lgraph = addLayers(lgraph, sequenceInputLayer(input_size, 'Name', 'input'));
lgraph = addLayers(lgraph, lstmLayer(64, 'OutputMode', 'sequence', 'Name', 'encoder_lstm'));
lgraph = addLayers(lgraph, selfAttentionLayer(1, 64, 'Name', 'attention_bahdanau'));
lgraph = addLayers(lgraph, lstmLayer(32, 'OutputMode', 'last', 'Name', 'decoder_lstm'));
lgraph = addLayers(lgraph, fullyConnectedLayer(1, 'Name', 'fc_out'));
lgraph = addLayers(lgraph, regressionLayer('Name', 'regression'));

lgraph = connectLayers(lgraph, 'input', 'encoder_lstm');
lgraph = connectLayers(lgraph, 'encoder_lstm', 'attention_bahdanau');
lgraph = connectLayers(lgraph, 'attention_bahdanau', 'decoder_lstm');
lgraph = connectLayers(lgraph, 'decoder_lstm', 'fc_out');
lgraph = connectLayers(lgraph, 'fc_out', 'regression');
end

function [X_all_cell, Y_all_mat] = build_all_sequences(curves, window_size, sequence_stride)
X_all_cell = {};
Y_all_mat = [];
for i = 1:numel(curves)
    u_c = curves(i).u;
    du_dt_c = curves(i).du_dt;
    M_c = curves(i).M;

    [X_c, Y_c] = create_sequences(u_c, du_dt_c, M_c, window_size, sequence_stride);
    for seq_idx = 1:size(X_c, 1)
        X_all_cell{end+1, 1} = squeeze(X_c(seq_idx, :, :))';
    end
    Y_all_mat = [Y_all_mat; Y_c];
end
end

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

function [M_pred, attention_series, attention_map] = closed_loop_with_occlusion_attention(net, curve, window_size)
u = curve.u;
du_dt = curve.du_dt;
M_true = curve.M;

N = numel(M_true);
M_pred = zeros(size(M_true));
M_pred(1:window_size) = M_true(1:window_size);

attention_acc = zeros(N, 1);
attention_count = zeros(N, 1);
attention_map = zeros(N, window_size);

for k = window_size + 1:N
    u_hist = u(k-window_size+1:k);
    du_hist = du_dt(k-window_size+1:k);
    M_hist = M_pred(k-window_size:k-1);
    X_step = [u_hist; du_hist; M_hist];

    M_k = predict(net, X_step, 'ExecutionEnvironment', 'cpu');
    M_pred(k) = M_k;

    local_w = estimate_window_attention_occlusion(net, X_step, M_k);
    attention_map(k, :) = local_w;

    abs_idx = (k-window_size+1):k;
    attention_acc(abs_idx) = attention_acc(abs_idx) + local_w(:);
    attention_count(abs_idx) = attention_count(abs_idx) + 1;
end

attention_series = zeros(N, 1);
valid = attention_count > 0;
attention_series(valid) = attention_acc(valid) ./ attention_count(valid);

if any(valid)
    min_w = min(attention_series(valid));
    max_w = max(attention_series(valid));
    if max_w > min_w
        attention_series(valid) = (attention_series(valid) - min_w) ./ (max_w - min_w);
    end
end
end

function local_w = estimate_window_attention_occlusion(net, X_step, base_pred)
window_size = size(X_step, 2);
local_w = zeros(1, window_size);

baseline = mean(X_step, 2);
for j = 1:window_size
    X_occ = X_step;
    X_occ(:, j) = baseline;
    pred_occ = predict(net, X_occ, 'ExecutionEnvironment', 'cpu');
    local_w(j) = abs(base_pred - pred_occ);
end

sum_w = sum(local_w);
if sum_w > 0
    local_w = local_w ./ sum_w;
else
    local_w(:) = 1 / window_size;
end
end

function idx = find_reversal_points(du_dt)
s = sign(du_dt(:));
idx = find( ...
    (s(1:end-1) .* s(2:end) <= 0) & ...
    (s(1:end-1) ~= s(2:end)) & ...
    ((s(1:end-1) ~= 0) | (s(2:end) ~= 0)) ...
    ) + 1;
end

function curve_id = choose_curve_with_most_reversals(test_curves)
num_curves = numel(test_curves);
rev_counts = zeros(num_curves, 1);
for i = 1:num_curves
    rev_counts(i) = numel(find_reversal_points(test_curves(i).du_dt));
end
[~, curve_id] = max(rev_counts);
end

function segment_idx = choose_segment_around_reversals(N, reversal_idx, window_size)
if isempty(reversal_idx)
    center = floor(N / 2);
else
    center = reversal_idx(min(2, numel(reversal_idx)));
end

half_span = max(80, 2 * window_size);
start_idx = max(1, center - half_span);
end_idx = min(N, center + half_span);
segment_idx = start_idx:end_idx;
end

function h = draw_reversal_lines(t, reversal_idx, style)
h = [];
if isempty(reversal_idx)
    return;
end
for i = 1:numel(reversal_idx)
    h_i = xline(t(reversal_idx(i)), style);
    if i == 1
        h_i.DisplayName = 'Reversal points';
        h = h_i;
    end
end
end

function add_legend_with_optional_reversals(h1, h2, h_rev)
if isempty(h_rev)
    legend([h1, h2], {'M true', 'M pred'}, 'Location', 'best');
else
    legend([h1, h2, h_rev], {'M true', 'M pred', 'Reversal points'}, 'Location', 'best');
end
end

function [trainedNet, tipo_levels] = train_tft_interpretability_model(train_curves, window_size, prediction_horizon, sequence_stride, val_split, mini_batch_size, max_epochs, tipo_levels, quantiles, num_hidden_units, num_attention_heads, num_categories, num_output_quantiles, dropout_probability)
M_past = [];
u_past_fut = [];
du_past_fut = [];
tipo_stat = [];
A1_stat = [];
A2_stat = [];
Y_target = [];

for i = 1:numel(train_curves)
    u_c = train_curves(i).u;
    du_dt_c = train_curves(i).du_dt;
    M_c = train_curves(i).M;
    ex_type = train_curves(i).type;
    A1 = train_curves(i).A1;
    A2 = train_curves(i).A2;

    [Xp_c, Xf_c, Xs_c, Y_c] = create_tft_sequences_local( ...
        u_c, du_dt_c, M_c, ex_type, A1, A2, window_size, prediction_horizon, sequence_stride);

    M_curve = permute(Xp_c(:, :, 1), [2, 3, 1]); % T x C x B
    M_past = cat(3, M_past, M_curve);

    u_pf_2d = [Xp_c(:, :, 2)'; Xf_c(:, :, 1)'];
    du_pf_2d = [Xp_c(:, :, 3)'; Xf_c(:, :, 2)'];
    u_pf = reshape(u_pf_2d, window_size + prediction_horizon, 1, size(Xp_c, 1));
    du_pf = reshape(du_pf_2d, window_size + prediction_horizon, 1, size(Xp_c, 1));

    u_past_fut = cat(3, u_past_fut, u_pf);
    du_past_fut = cat(3, du_past_fut, du_pf);
    tipo_stat = [tipo_stat; Xs_c(:, 1)];
    A1_stat = [A1_stat; Xs_c(:, 2)];
    A2_stat = [A2_stat; Xs_c(:, 3)];

    Y_curve = permute(Y_c(:, :, 1), [2, 3, 1]);
    Y_target = cat(3, Y_target, Y_curve);
end

n_total = size(M_past, 3);
perm_idx = randperm(n_total);
n_val = floor(val_split * n_total);
val_idx = perm_idx(1:n_val);
train_idx = perm_idx(n_val + 1:end);

val_idx = val_idx(1:floor(numel(val_idx) / mini_batch_size) * mini_batch_size);
train_idx = train_idx(1:floor(numel(train_idx) / mini_batch_size) * mini_batch_size);

tipo_cat_all = categorical(tipo_stat, [1 2 3 4 5 6 7], tipo_levels);

if isempty(train_idx) || isempty(val_idx)
    trainedNet = [];
    return;
end

dsTrain = create_tft_multi_input_datastore_local( ...
    M_past(:, :, train_idx), u_past_fut(:, :, train_idx), du_past_fut(:, :, train_idx), ...
    tipo_cat_all(train_idx), A1_stat(train_idx), A2_stat(train_idx), Y_target(:, :, train_idx));

dsVal = create_tft_multi_input_datastore_local( ...
    M_past(:, :, val_idx), u_past_fut(:, :, val_idx), du_past_fut(:, :, val_idx), ...
    tipo_cat_all(val_idx), A1_stat(val_idx), A2_stat(val_idx), Y_target(:, :, val_idx));

inputNames = ["M", "u", "du_dt", "tipo", "A1", "A2"];
netTFT = createTFTNetwork(inputNames, 1, [2 3], [4 5 6], 4, num_categories, num_hidden_units, ...
    num_attention_heads, window_size, prediction_horizon, num_output_quantiles, DropoutProbability=dropout_probability);

options = trainingOptions("adam", ...
    MaxEpochs=max_epochs, ...
    MiniBatchSize=mini_batch_size, ...
    ExecutionEnvironment="cpu", ...
    InitialLearnRate=0.001, ...
    GradientThreshold=0.01, ...
    Shuffle="every-epoch", ...
    ValidationData=dsVal, ...
    ValidationFrequency=100, ...
    Plots="training-progress", ...
    Verbose=true);

trainedNet = trainnet(dsTrain, netTFT, @(Y,T) quantile_loss_local(Y, T, quantiles, DataFormat="CBT"), options);
end

function [M_pred, attention_series, attention_map] = closed_loop_tft_with_occlusion_attention(trainedNet, curve, window_size, prediction_horizon, tipo_levels)
u = curve.u(:);
du_dt = curve.du_dt(:);
M_true = curve.M(:);
N = numel(M_true);

M_pred = M_true;
attention_acc = zeros(N, 1);
attention_count = zeros(N, 1);
attention_map = zeros(N, window_size + prediction_horizon);

type_c = categorical(curve.type, [1 2 3 4 5 6 7], tipo_levels);
A1_c = curve.A1;
A2_c = curve.A2;

for k = window_size + 1:prediction_horizon:(N - prediction_horizon + 1)
    u_past = u(k-window_size:k-1);
    du_past = du_dt(k-window_size:k-1);
    M_past = M_pred(k-window_size:k-1);
    u_fut = u(k:k+prediction_horizon-1);
    du_fut = du_dt(k:k+prediction_horizon-1);

    pred_quantiles = predict_tft_quantiles(trainedNet, M_past, u_past, du_past, u_fut, du_fut, type_c, A1_c, A2_c, prediction_horizon);
    M_block = pred_quantiles(:, 2);
    M_pred(k:k+prediction_horizon-1) = M_block;

    local_w = estimate_tft_window_attention_occlusion( ...
        trainedNet, M_past, u_past, du_past, u_fut, du_fut, type_c, A1_c, A2_c, prediction_horizon, M_block);
    attention_map(k, :) = local_w;

    abs_idx = (k-window_size):(k+prediction_horizon-1);
    attention_acc(abs_idx) = attention_acc(abs_idx) + local_w(:);
    attention_count(abs_idx) = attention_count(abs_idx) + 1;

    if mod(k - (window_size + 1), 5 * prediction_horizon) == 0
        fprintf('  TFT attention progress: k=%d/%d\n', k, N - prediction_horizon + 1);
    end
end

attention_series = zeros(N, 1);
valid = attention_count > 0;
attention_series(valid) = attention_acc(valid) ./ attention_count(valid);
if any(valid)
    min_w = min(attention_series(valid));
    max_w = max(attention_series(valid));
    if max_w > min_w
        attention_series(valid) = (attention_series(valid) - min_w) ./ (max_w - min_w);
    end
end
end

function pred_quantiles = predict_tft_quantiles(trainedNet, M_past, u_past, du_past, u_fut, du_fut, type_c, A1_c, A2_c, prediction_horizon)
M_tensor = reshape(double(M_past(:)), numel(M_past), 1, 1);
u_tensor = reshape(double([u_past(:); u_fut(:)]), numel(u_past) + numel(u_fut), 1, 1);
du_tensor = reshape(double([du_past(:); du_fut(:)]), numel(du_past) + numel(du_fut), 1, 1);
tgt_tensor = zeros(prediction_horizon, 1, 1);

tmp_ds = create_tft_multi_input_datastore_local(M_tensor, u_tensor, du_tensor, type_c, A1_c, A2_c, tgt_tensor);
pred_quantiles = minibatchpredict(trainedNet, tmp_ds, 'Outputs', 'quantile_out');
end

function local_w = estimate_tft_window_attention_occlusion(trainedNet, M_past, u_past, du_past, u_fut, du_fut, type_c, A1_c, A2_c, prediction_horizon, base_block)
window_size = numel(M_past);
total_steps = window_size + prediction_horizon;
local_w = zeros(total_steps, 1);

base_summary = mean(base_block, 'omitnan');

u_all = [u_past(:); u_fut(:)];
du_all = [du_past(:); du_fut(:)];
baseline_u = mean(u_all, 'omitnan');
baseline_du = mean(du_all, 'omitnan');
baseline_m = mean(M_past, 'omitnan');

for j = 1:total_steps
    u_occ = u_all;
    du_occ = du_all;
    M_occ = M_past;

    u_occ(j) = baseline_u;
    du_occ(j) = baseline_du;
    if j <= window_size
        M_occ(j) = baseline_m;
    end

    u_past_occ = u_occ(1:window_size);
    u_fut_occ = u_occ(window_size+1:end);
    du_past_occ = du_occ(1:window_size);
    du_fut_occ = du_occ(window_size+1:end);

    pred_occ = predict_tft_quantiles( ...
        trainedNet, M_occ, u_past_occ, du_past_occ, u_fut_occ, du_fut_occ, type_c, A1_c, A2_c, prediction_horizon);
    local_w(j) = abs(base_summary - mean(pred_occ(:, 2), 'omitnan'));
end

sum_w = sum(local_w);
if sum_w > 0
    local_w = local_w ./ sum_w;
else
    local_w(:) = 1 / total_steps;
end
end

function [X_past, X_fut, X_stat, Y_tgt] = create_tft_sequences_local(u, du_dt, M, tipo, A1, A2, w_size, p_horizon, stride)
if length(u) <= (w_size + p_horizon)
    warning('Sequence length %d is insufficient (minimum required: %d for window size %d + prediction horizon %d).', ...
        length(u), w_size + p_horizon + 1, w_size, p_horizon);
    X_past = zeros(0, w_size, 3);
    X_fut = zeros(0, p_horizon, 2);
    X_stat = zeros(0, 3);
    Y_tgt = zeros(0, p_horizon, 1);
    return;
end
start_indices = 1:stride:(length(u) - w_size - p_horizon);
n_samples = numel(start_indices);

X_past = zeros(n_samples, w_size, 3);
X_fut = zeros(n_samples, p_horizon, 2);
X_stat = zeros(n_samples, 3);
Y_tgt = zeros(n_samples, p_horizon, 1);

for i = 1:n_samples
    start_idx = start_indices(i);
    curr_idx = start_idx + w_size - 1;

    X_past(i, :, 1) = M(start_idx:curr_idx);
    X_past(i, :, 2) = u(start_idx:curr_idx);
    X_past(i, :, 3) = du_dt(start_idx:curr_idx);

    fut_start = curr_idx + 1;
    fut_end = curr_idx + p_horizon;
    X_fut(i, :, 1) = u(fut_start:fut_end);
    X_fut(i, :, 2) = du_dt(fut_start:fut_end);

    X_stat(i, :) = [tipo, A1, A2];
    Y_tgt(i, :, 1) = M(fut_start:fut_end);
end
end

function ds = create_tft_multi_input_datastore_local(M_p, u_pf, du_pf, tipo_cat, A1_s, A2_s, Y_tgt_c)
M_p = ensure_tcb_single_channel_local(M_p);
u_pf = ensure_tcb_single_channel_local(u_pf);
du_pf = ensure_tcb_single_channel_local(du_pf);
Y_tgt_c = ensure_tcb_single_channel_local(Y_tgt_c);

ads_M = arrayDatastore(M_p, 'IterationDimension', 3);
ads_u = arrayDatastore(u_pf, 'IterationDimension', 3);
ads_du = arrayDatastore(du_pf, 'IterationDimension', 3);
ads_tipo = arrayDatastore(tipo_cat);
ads_A1 = arrayDatastore(double(A1_s(:)));
ads_A2 = arrayDatastore(double(A2_s(:)));
adsTarget = arrayDatastore(Y_tgt_c, 'IterationDimension', 3);
ds = combine(ads_M, ads_u, ads_du, ads_tipo, ads_A1, ads_A2, adsTarget);
end

function X = ensure_tcb_single_channel_local(X)
X = double(X);
if isvector(X)
    X = reshape(X(:), numel(X), 1, 1);
elseif ismatrix(X)
    X = reshape(X, size(X, 1), 1, size(X, 2));
end

if ndims(X) > 3
    error('Expected 3D tensor (T x C x B), but received a %d-dimensional tensor.', ndims(X));
end
if size(X, 2) > 1 && size(X, 3) == 1
    X = permute(X, [1 3 2]);
end
if size(X, 2) ~= 1
    error('Channel dimension must be 1, got %d.', size(X, 2));
end
end

function l = quantile_loss_local(Y, T, quantiles, options)
arguments
    Y
    T
    quantiles
    options.DataFormat = "TCB"
end
predictionUnderflow = T - Y;
channelDim = strfind(options.DataFormat, "C");
quantiles_sh = shiftdim(quantiles, 1 - channelDim);
qLoss = quantiles_sh .* max(predictionUnderflow, 0) + (1 - quantiles_sh) .* max(-predictionUnderflow, 0);
observationDim = strfind(options.DataFormat, "B");
timeDim = strfind(options.DataFormat, "T");
l = sum(qLoss, "all") / (size(Y, observationDim) * size(Y, timeDim));
end
