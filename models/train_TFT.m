% Modello globale TFT (Temporal Fusion Transformer) per predizione cicli di isteresi
% Utilizza i dati generati centralmente dal modulo hysteresis_training.m
% Richiede MATLAB R2025a per le funzioni createTFTNetwork e helper nativi.

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
%  2. PREPARAZIONE SEQUENZE PER TFT
% ============================================================
fprintf('\n=== PREPARAZIONE DATI TFT ===\n');

window_size        = 100;   % Finestra temporale di osservazione (passato)
prediction_horizon = 25;   % Orizzonte di predizione in avanti (futuro)
sequence_stride    = 20;
val_split   = 0.2;  

M_past       = []; % [M] passi passati (Unknown)
u_past_fut   = []; % [u] past + future (Known)
du_ast_fut   = []; % [du/dt] past + future (Known)
tipo_stat    = []; % [tipo] categorical
A1_stat      = []; % [A1] static numeric
A2_stat      = []; % [A2] static numeric
Y_target_cell= []; % [M] passi futuri target

for i = 1:numel(train_curves)
    u_c = train_curves(i).u;
    du_dt_c = train_curves(i).du_dt;
    M_c = train_curves(i).M;
    ex_type = train_curves(i).type;
    A1 = train_curves(i).A1;
    A2 = train_curves(i).A2;
    
    [Xp_c, Xf_c, Xs_c, Y_c] = create_tft_sequences(u_c, du_dt_c, M_c, ...
        ex_type, A1, A2, window_size, prediction_horizon, sequence_stride);

    % Concateniamo nella terza dimensione (Observazioni) nel formato T x C x B
    M_curve = permute(Xp_c(:, :, 1), [2, 3, 1]); % window_size x 1 x n_samples
    M_past = cat(3, M_past, M_curve);
    
    % Known time-varying necessitano passato e futuro concatenati
    u_pf_2d  = [Xp_c(:, :, 2)'; Xf_c(:, :, 1)'];
    du_pf_2d = [Xp_c(:, :, 3)'; Xf_c(:, :, 2)'];
    u_pf = reshape(u_pf_2d, window_size + prediction_horizon, 1, size(Xp_c, 1));
    du_pf = reshape(du_pf_2d, window_size + prediction_horizon, 1, size(Xp_c, 1));
    
    u_past_fut   = cat(3, u_past_fut, u_pf);
    du_ast_fut   = cat(3, du_ast_fut, du_pf);
    
    tipo_stat    = [tipo_stat; Xs_c(:, 1)];
    A1_stat      = [A1_stat; Xs_c(:, 2)];
    A2_stat      = [A2_stat; Xs_c(:, 3)];
    
    Y_curve = permute(Y_c(:, :, 1), [2, 3, 1]); % prediction_horizon x 1 x n_samples
    Y_target_cell = cat(3, Y_target_cell, Y_curve);
end

n_total = size(M_past, 3);
rng(42, 'twister');
perm_idx = randperm(n_total);
n_val = floor(val_split * n_total);
val_idx = perm_idx(1:n_val);
train_idx = perm_idx(n_val + 1:end);

% Tronchiamo i set affinché siano multipli esatti del MiniBatchSize (64)
MINI_BATCH_SIZE = 64;
val_idx = val_idx(1 : floor(numel(val_idx)/MINI_BATCH_SIZE)*MINI_BATCH_SIZE);
train_idx = train_idx(1 : floor(numel(train_idx)/MINI_BATCH_SIZE)*MINI_BATCH_SIZE);

% Attenzione: mappare i Tipi Esatti generati (1, 2, 3, 4, 5, 6, 7)
tipo_cat_all = categorical(tipo_stat, [1 2 3 4 5 6 7], {'Multi-Sine', 'Damped Sine', 'Stepwise', 'Triangular', 'Concentric Loops', 'FORC', 'Dense Minor Loops'});

dsTrain = create_tft_multi_input_datastore(M_past(:, :, train_idx), u_past_fut(:, :, train_idx), du_ast_fut(:, :, train_idx), ...
    tipo_cat_all(train_idx), A1_stat(train_idx), A2_stat(train_idx), Y_target_cell(:, :, train_idx));
    
dsVal   = create_tft_multi_input_datastore(M_past(:, :, val_idx), u_past_fut(:, :, val_idx), du_ast_fut(:, :, val_idx), ...
    tipo_cat_all(val_idx), A1_stat(val_idx), A2_stat(val_idx), Y_target_cell(:, :, val_idx));

fprintf('Split TFT -> training: %d sequenze | validation: %d sequenze\n', numel(train_idx), numel(val_idx));

%% ============================================================
%  3. TFT ARCHITECTURE & TRAINING
% ============================================================

inputNames = ["M", "u", "du_dt", "tipo", "A1", "A2"];

unknownTimeVaryingInputIdx = 1;      
knownTimeVaryingInputIdx   = [2 3];  
staticInputIdx             = [4 5 6];
categoricalInputIdx        = 4;      

numCategories       = 7;   % (Multi-Sine, Damped Sine, Stepwise, Triangular, Concentric Loops, FORC, Dense)
numHiddenUnits      = 64;  
numAttentionHeads   = 8;   
numQuantiles        = 3;   
dropoutProbability  = 0.1;
quantiles = [0.1; 0.5; 0.9];

if exist('createTFTNetwork', 'file')
    netTFT = createTFTNetwork(inputNames, unknownTimeVaryingInputIdx, ...
        knownTimeVaryingInputIdx, staticInputIdx, categoricalInputIdx, ...
        numCategories, numHiddenUnits, numAttentionHeads, window_size, ...
        prediction_horizon, numQuantiles, ...
        DropoutProbability=dropoutProbability);

    options = trainingOptions("adam", ...
        MaxEpochs=100, ...
        MiniBatchSize=MINI_BATCH_SIZE, ...
        ExecutionEnvironment="cpu", ...
        InitialLearnRate=0.001, ...
        GradientThreshold=0.01, ...
        Shuffle="every-epoch", ...
        Plots="training-progress", ...
        Verbose=true);

    fprintf('\n=== ADDESTRAMENTO MODELLO TFT (R2025a) ===\n');
    trainedNet = trainnet(dsTrain, netTFT, @(Y,T) quantileLoss(Y, T, quantiles, DataFormat="CBT"), options);
else
    fprintf('\n[ATTENZIONE] Funzione createTFTNetwork non trovata (richiesta R2025a).\n');
    trainedNet = [];
end

%% ============================================================
%  4. INFERENZA MULTI-HORIZON SUL TEST SET
% ============================================================

if ~isempty(trainedNet)
    fprintf('\n=== VALUTAZIONE TEST SET GLOBALE (TFT CLOSED-LOOP) ===\n');
    fprintf('%-5s %-13s %-10s %-10s %-8s %-12s %-12s\n', ...
        'ID', 'Tipo', 'RMSE', 'MAE', 'R^2', 'Area Err %', 'CoerciveErr');

    results_tft = struct('test_id', {}, 'type_name', {}, 'mse', {}, 'rmse', {}, ...
        'mae', {}, 'r2', {}, 'area_err_pct', {}, 'coercive_err', {});

    for test_id = 1:numel(test_curves)
        u_c = test_curves(test_id).u;
        du_dt_c = test_curves(test_id).du_dt;
        M_c = test_curves(test_id).M;
        type_c = categorical(test_curves(test_id).type, [1 2 3 4 5 6 7], {'Multi-Sine', 'Damped Sine', 'Stepwise', 'Triangular', 'Concentric Loops', 'FORC', 'Dense Minor Loops'});
        A1_c   = test_curves(test_id).A1;
        A2_c   = test_curves(test_id).A2;
        
        M_nn_c_q10 = zeros(size(M_c));
        M_nn_c_q50 = zeros(size(M_c));
        M_nn_c_q90 = zeros(size(M_c));
        
        M_nn_c_q10(1:window_size) = M_c(1:window_size);
        M_nn_c_q50(1:window_size) = M_c(1:window_size);
        M_nn_c_q90(1:window_size) = M_c(1:window_size);
        
        for k = window_size+1 : prediction_horizon : (length(M_c) - prediction_horizon + 1)
            u_past  = u_c(k-window_size:k-1);
            du_past = du_dt_c(k-window_size:k-1);
            M_past  = M_nn_c_q50(k-window_size:k-1); 
            
            u_fut  = u_c(k:k+prediction_horizon-1);
            du_fut = du_dt_c(k:k+prediction_horizon-1);
            
            M_tensor = reshape(double(M_past(:)), window_size, 1, 1);
            u_tensor = reshape(double([u_past(:); u_fut(:)]), window_size + prediction_horizon, 1, 1);
            du_tensor = reshape(double([du_past(:); du_fut(:)]), window_size + prediction_horizon, 1, 1);
            tgt_tensor = zeros(prediction_horizon, 1, 1);
            
            tmp_ds = create_tft_multi_input_datastore(...
                M_tensor, u_tensor, du_tensor, ...
                type_c, A1_c, A2_c, tgt_tensor);
            
            pred_quantiles = minibatchpredict(trainedNet, tmp_ds, 'Outputs', 'quantile_out'); 
            
            M_nn_c_q10(k:k+prediction_horizon-1) = pred_quantiles(:,1);
            M_nn_c_q50(k:k+prediction_horizon-1) = pred_quantiles(:,2);
            M_nn_c_q90(k:k+prediction_horizon-1) = pred_quantiles(:,3);
        end
        
        % --- Calcolo Metriche (sulla mediana q50) ---
        idx_valid = window_size+1:length(M_c);
        err = M_c(idx_valid) - M_nn_c_q50(idx_valid);
        
        % RMSE = sqrt(mean((y - yhat).^2))
        mse_c  = mean(err.^2);
        rmse_c = sqrt(mse_c);
        % MAE = mean(abs(y - yhat))
        mae_c  = mean(abs(err));
        
        % R^2 = 1 - sum((y - yhat).^2) / sum((y - mean(y)).^2)
        denom_r2 = sum((M_c(idx_valid) - mean(M_c(idx_valid))).^2);
        if denom_r2 > 0
            r2_c = 1 - sum(err.^2) / denom_r2;
        else
            r2_c = NaN;
        end
        
        % Area error %% = 100 * abs(|int y du| - |int yhat du|) / |int y du|
        area_preisach = abs(trapz(u_c, M_c));
        area_tft      = abs(trapz(u_c, M_nn_c_q50));
        if area_preisach > 0
            area_err_pct = abs(area_tft - area_preisach) / area_preisach * 100;
        else
            area_err_pct = NaN;
        end

        % CoerciveErr = mean(|Hc-_pred-Hc-_true|, |Hc+_pred-Hc+_true|)
        [hc_neg_true, hc_pos_true] = estimate_coercive_fields(u_c, M_c);
        [hc_neg_pred, hc_pos_pred] = estimate_coercive_fields(u_c, M_nn_c_q50);
        coercive_err_c = mean(abs([hc_neg_pred - hc_neg_true, hc_pos_pred - hc_pos_true]), 'omitnan');

        results_tft(test_id).test_id      = test_id;
        results_tft(test_id).type_name    = test_curves(test_id).name;
        results_tft(test_id).mse          = mse_c;
        results_tft(test_id).rmse         = rmse_c;
        results_tft(test_id).mae          = mae_c;
        results_tft(test_id).r2           = r2_c;
        results_tft(test_id).area_err_pct = area_err_pct;
        results_tft(test_id).coercive_err = coercive_err_c;

        test_curves(test_id).M_pred_q10 = M_nn_c_q10;
        test_curves(test_id).M_pred_q50 = M_nn_c_q50;
        test_curves(test_id).M_pred_q90 = M_nn_c_q90;

        fprintf('%-5d %-13s %-10.4e %-10.4e %-8.4f %-12.2f %-12.4e\n', ...
            test_id, test_curves(test_id).name, rmse_c, mae_c, r2_c, area_err_pct, coercive_err_c);
    end
end
    %% ============================================================
    %  5. VISUALIZZAZIONE RISULTATI
    % ============================================================
    unique_types = unique({test_curves.name});

    for i = 1:numel(unique_types)
        type_name = unique_types{i};
        idx_type = find(strcmp({test_curves.name}, type_name));

        figure('Position', [60, 80, 1450, 650], ...
            'Name', sprintf('Test TFT Closed-Loop Quantiles - %s', type_name));

        for local_plot_idx = 1:numel(idx_type)
            curve_idx = idx_type(local_plot_idx);
            u_c = test_curves(curve_idx).u;
            M_c = test_curves(curve_idx).M;
            t_c = test_curves(curve_idx).t;
            
            M_q10 = test_curves(curve_idx).M_pred_q10(:);
            M_q50 = test_curves(curve_idx).M_pred_q50(:);
            M_q90 = test_curves(curve_idx).M_pred_q90(:);
            
            u_c = u_c(:);
            t_c = t_c(:);

            subplot(2, numel(idx_type), local_plot_idx);
            plot(u_c, M_c, 'b-', 'LineWidth', 1.5); hold on;
            patch([u_c; flipud(u_c)], [M_q10; flipud(M_q90)], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
            plot(u_c, M_q50, 'r--', 'LineWidth', 1.5);
            xlabel('u(t)', 'FontSize', 11);
            ylabel('M(t)', 'FontSize', 11);
            title(sprintf('%s Test %d\nR^2=%.4f | \\DeltaArea=%.2f%%', ...
                type_name, local_plot_idx, ...
                results_tft(curve_idx).r2, results_tft(curve_idx).area_err_pct), 'FontSize', 11);
            legend('Preisach', '90% Conf.', 'TFT (q=0.5)', 'Location', 'best', 'FontSize', 9);
            grid on;

            subplot(2, numel(idx_type), numel(idx_type) + local_plot_idx);
            plot(t_c, M_c, 'b-', 'LineWidth', 1.5); hold on;
            patch([t_c; flipud(t_c)], [M_q10; flipud(M_q90)], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
            plot(t_c, M_q50, 'r--', 'LineWidth', 1.5);
            xlabel('Time [s]', 'FontSize', 11);
            ylabel('M(t)', 'FontSize', 11);
            title(sprintf('RMSE=%.4e | MAE=%.4e', ...
                results_tft(curve_idx).rmse, results_tft(curve_idx).mae), 'FontSize', 11);
            legend('Preisach', '90% Conf.', 'TFT', 'Location', 'best', 'FontSize', 9);
            xlim([0, t_c(end)]);
            grid on;
        end
    end

    fprintf('\n=== SUMMARY BY EXCITATION TYPE (TFT) ===\n');
    for i = 1:numel(unique_types)
        type_name = unique_types{i};
        idx_type = find(strcmp({results_tft.type_name}, type_name));
        
        mean_r2 = mean([results_tft(idx_type).r2], 'omitnan');
        mean_area_err = mean([results_tft(idx_type).area_err_pct], 'omitnan');
        fprintf('%-13s | Mean R^2 = %.4f | Mean Area Error = %.2f%%\n', ...
            type_name, mean_r2, mean_area_err);
    end

    % --- SALVATAGGIO RISULTATI ---
    timestamp = datestr(now, 'yyyymmdd_HHMM');
    save_folder = fullfile(repo_root, 'results_comparison');
    if ~exist(save_folder, 'dir'), mkdir(save_folder); end
    % 1. Salvataggio struct dei risultati e delle curve (.mat)
    save(fullfile(save_folder, ['results_TFT_' timestamp '.mat']), 'results_tft', 'test_curves');
    
    txt_filename = fullfile(save_folder, ['summary_TFT_' timestamp '.txt']);
    fid = fopen(txt_filename, 'w');
    if fid ~= -1
        fprintf(fid, '=== RIEPILOGO PERFORMANCE TFT (%s) ===\n\n', datestr(now));
        fprintf(fid, '%-5s %-13s %-10s %-10s %-8s %-12s %-12s\n', ...
            'ID', 'Type', 'RMSE', 'MAE', 'R^2', 'Area Err %', 'CoerciveErr');
        for test_id = 1:numel(results_tft)
            fprintf(fid, '%-5d %-13s %-10.4e %-10.4e %-8.4f %-12.2f %-12.4e\n', ...
                test_id, results_tft(test_id).type_name, results_tft(test_id).rmse, ...
                results_tft(test_id).mae, results_tft(test_id).r2, ...
                results_tft(test_id).area_err_pct, results_tft(test_id).coercive_err);
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
            idx_type = find(strcmp({results_tft.type_name}, type_name));
            fprintf(fid, '%-13s | Mean R^2 = %.4f | Mean Area Error = %.2f%%\n', ...
                type_name, mean([results_tft(idx_type).r2], 'omitnan'), ...
                mean([results_tft(idx_type).area_err_pct], 'omitnan'));
        end
        
        fprintf(fid, '\n--- In-Domain Average (Types 1, 5, 6, 7) ---\n');
        in_domain_types = {'Multi-Sine', 'Concentric Loops', 'FORC', 'Dense Minor Loops'};
        idx_in = find(ismember({results_tft.type_name}, in_domain_types));
        fprintf(fid, 'Mean R^2 = %.4f | Mean Area Error = %.2f%%\n', ...
                mean([results_tft(idx_in).r2], 'omitnan'), mean([results_tft(idx_in).area_err_pct], 'omitnan'));
                
        fprintf(fid, '\n--- Extended Average (All Types) ---\n');
        fprintf(fid, 'Mean R^2 = %.4f | Mean Area Error = %.2f%%\n', ...
                mean([results_tft.r2], 'omitnan'), mean([results_tft.area_err_pct], 'omitnan'));

        fclose(fid);
        fprintf('\nRisultati salvati in: %s\n', save_folder);
    end
%end

%% ============================================================
%  LOCAL HELPER FUNCTIONS DI SUPPORTO
% ============================================================

function [X_past, X_fut, X_stat, Y_tgt] = create_tft_sequences(u, du_dt, M, tipo, A1, A2, w_size, p_horizon, stride)
    start_indices = 1:stride:(length(u) - w_size - p_horizon);
    n_samples = numel(start_indices);
    
    X_past = zeros(n_samples, w_size, 3);
    X_fut  = zeros(n_samples, p_horizon, 2);
    X_stat = zeros(n_samples, 3);
    Y_tgt  = zeros(n_samples, p_horizon, 1);

    for i = 1:n_samples
        start_idx = start_indices(i);
        curr_idx = start_idx + w_size - 1;
        
        X_past(i, :, 1) = M(start_idx:curr_idx);
        X_past(i, :, 2) = u(start_idx:curr_idx);
        X_past(i, :, 3) = du_dt(start_idx:curr_idx);
        
        fut_start = curr_idx + 1;
        fut_end   = curr_idx + p_horizon;
        
        X_fut(i, :, 1) = u(fut_start:fut_end);
        X_fut(i, :, 2) = du_dt(fut_start:fut_end);
        
        X_stat(i, :) = [tipo, A1, A2];
        Y_tgt(i, :, 1) = M(fut_start:fut_end);
    end
end

function ds = create_tft_multi_input_datastore(M_p, u_pf, du_pf, tipo_cat, A1_s, A2_s, Y_tgt_c)
    M_p = ensure_tcb_single_channel(M_p, "M");
    u_pf = ensure_tcb_single_channel(u_pf, "u");
    du_pf = ensure_tcb_single_channel(du_pf, "du_dt");
    Y_tgt_c = ensure_tcb_single_channel(Y_tgt_c, "target");
    
    ads_M    = arrayDatastore(M_p, 'IterationDimension', 3);
    ads_u    = arrayDatastore(u_pf, 'IterationDimension', 3);
    ads_du   = arrayDatastore(du_pf, 'IterationDimension', 3);
    
    ads_tipo = arrayDatastore(tipo_cat);
    ads_A1   = arrayDatastore(double(A1_s(:)));
    ads_A2   = arrayDatastore(double(A2_s(:)));
    adsTarget = arrayDatastore(Y_tgt_c, 'IterationDimension', 3);
    
    ds = combine(ads_M, ads_u, ads_du, ads_tipo, ads_A1, ads_A2, adsTarget);
end

function X = ensure_tcb_single_channel(X, ~)
    X = double(X);
    if isvector(X), X = reshape(X(:), numel(X), 1, 1);
    elseif ismatrix(X), X = reshape(X, size(X, 1), 1, size(X, 2)); end
    
    if ndims(X) > 3, error('Atteso tensore 3D T x C x B.'); end
    if size(X, 2) > 1 && size(X, 3) == 1, X = permute(X, [1 3 2]); end
    if size(X, 2) ~= 1, error('Canale C deve essere 1.'); end
end

function l = quantileLoss(Y,T,quantiles,options)
arguments
    Y
    T
    quantiles
    options.DataFormat = "TCB"
end
    predictionUnderflow = T - Y;
    channelDim = strfind(options.DataFormat,"C");
    quantiles_sh = shiftdim(quantiles,1-channelDim);
    qLoss = quantiles_sh .* max(predictionUnderflow,0) + (1 - quantiles_sh) .* max(-predictionUnderflow,0);
    observationDim = strfind(options.DataFormat,"B");
    timeDim = strfind(options.DataFormat,"T");
    l = sum(qLoss,"all") / (size(Y,observationDim)*size(Y,timeDim));
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
