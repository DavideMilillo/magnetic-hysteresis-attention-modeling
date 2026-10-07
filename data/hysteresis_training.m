function [train_curves, test_curves] = hysteresis_training()
    % HYSTERESIS_TRAINING Genera il dataset globale per l'addestramento LSTM e TFT
    % Training (40 curve): 10x tipo1, 10x Tipo 5, 10x Tipo 6 (5 up, 5 down), 10x Tipo 7
    % Test (7 curve): 1x per tipo da 1 a 7 per verificare la generalizzazione
    
    fprintf('\n=== GENERAZIONE DATASET PREISACH (TRAINING + TEST) ===\n');
    
    rng(42, 'twister'); % Per riproducibilità
    
    % Parametri temporali globali
    T       = 0.7;      % Periodo fondamentale [s]
    f1      = 1 / T;    % Frequenza fondamentale [Hz]
    t_final = 4 * T;    % Durata simulazione [s]
    n_time_points = 2000; 
    dt      = t_final / (n_time_points - 1);
    
    %% 1. GRIGLIA DI PREISACH
    mu     = @(alpha, beta) exp(-(alpha.^2 + beta.^2)) .* (alpha >= beta);
    N_grid = 101;
    alpha  = linspace(-1, 1, N_grid);
    beta   = linspace(-1, 1, N_grid);
    [Alpha, Beta] = meshgrid(alpha, beta);
    P = mu(Alpha, Beta);
    P(Alpha <= Beta) = 0;
    
    %% 2. GENERAZIONE TRAINING SET (40 Curve)
    fprintf('\n--- Generazione Training Dataset (40 Curve: Tipi 1, 5, 6, 7) ---\n');
    train_curves = [];
    
    % --- 10 curve Tipo 1 (Multi-Sine) ---
    for i = 1:10
        cfg = struct('type', 1, 'A1', 0.75 + 0.25*rand(), 'A2', 0.15 + 0.35*rand(), ...
            'f2_ratio', randi([3, 7]), 'phi', rand()*2*pi, 'tau', NaN, 'step_levels', []);
        train_curves = [train_curves, generate_single_curve(cfg, f1, t_final, dt, Alpha, Beta, P, alpha)];
    end

    % --- 10 curve Tipo 5 (Concentric Loops) ---
    for i = 1:10
        cfg = struct('type', 5, 'A1', 1, 'A2', 0, 'f2_ratio', 1, 'phi', 0, 'tau', NaN, 'step_levels', []);
        cfg.concentric_cycles = randi([3, 7]);
        cfg.concentric_min_A = 0.2 + 0.4 * rand();
        cfg.concentric_max_A = 1.0;
        train_curves = [train_curves, generate_single_curve(cfg, f1, t_final, dt, Alpha, Beta, P, alpha)];
    end
    
    % --- 10 curve Tipo 6 (FORC: 5 Up, 5 Down) ---
    for i = 1:10
        cfg = struct('type', 6, 'A1', 1, 'A2', 0, 'f2_ratio', 1, 'phi', 0, 'tau', NaN, 'step_levels', []);
        cfg.forc_steps = 8;
        if i <= 5
            cfg.forc_direction = 'top'; 
        else
            cfg.forc_direction = 'bottom'; 
        end
        train_curves = [train_curves, generate_single_curve(cfg, f1, t_final, dt, Alpha, Beta, P, alpha)];
    end
    
    % --- 10 curve Tipo 7 (Dense Minor Loops) ---
    for i = 1:10
        cfg = struct('type', 7, 'A1', 1.0, 'A2', 0, 'f2_ratio', 1, 'phi', rand()*2*pi, 'tau', NaN, 'step_levels', []);
        cfg.dense_f_ratios = 13 + (31-13)*rand(); 
        cfg.dense_amps = 0.2 + (0.6-0.2)*rand();   
        train_curves = [train_curves, generate_single_curve(cfg, f1, t_final, dt, Alpha, Beta, P, alpha)];
    end
    
    %% 3. GENERAZIONE TEST SET (7 Curve)
    fprintf('\n--- Generazione Test Dataset (7 Curve: Tipi 1-7) ---\n');
    test_curves = [];
    for t_type = 1:7
        cfg = build_test_cfg(t_type, T, f1);
        test_curves = [test_curves, generate_single_curve(cfg, f1, t_final, dt, Alpha, Beta, P, alpha)];
    end
    
    fprintf('Generazione completata: %d train, %d test.\n', numel(train_curves), numel(test_curves));
end

function curve = generate_single_curve(cfg, f1, t_final, dt, Alpha, Beta, P, alpha)
    [u, M, t] = simulate_preisach(cfg, f1, t_final, dt, Alpha, Beta, P, alpha);
    du_dt = gradient(u, dt);
    du_dt = du_dt / (max(abs(du_dt)) + 1e-9); 
    
    curve = struct('type', cfg.type, 'name', get_excitation_name(cfg.type), ...
        'u', u, 'M', M, 't', t, 'du_dt', du_dt, 'A1', cfg.A1, 'A2', cfg.A2);
end

function cfg = build_test_cfg(type, T, f1)
    cfg = struct('type', type, 'f2_ratio', 5, 'A1', 0.8, 'A2', 0.4, 'phi', pi/4, 'tau', 1.5*T, ...
                 'step_levels', [0, 0.5, -0.4, 0.8, -0.7, 0.3, 0], 'concentric_cycles', 4, ...
                 'concentric_min_A', 0.4, 'concentric_max_A', 1.0, 'forc_steps', 6, 'forc_direction', 'top', ...
                 'dense_f_ratios', 25, 'dense_amps', 0.4, 'triangle_amplitude', 0.9, 'triangle_frequency', f1);
end

function [u, M, t] = simulate_preisach(cfg, f1, t_final, dt, Alpha, Beta, P, alpha)
    t = 0:dt:t_final;
    
    switch cfg.type
        case 1 % Multi-Sine
            u = cfg.A1 * sin(2*pi*f1*t) + cfg.A2 * sin(2*pi*cfg.f2_ratio*f1*t + cfg.phi);
        case 2 % Damped Sine
            u = exp(-t/cfg.tau) .* (cfg.A1 * sin(2*pi*f1*t) + cfg.A2 * sin(2*pi*cfg.f2_ratio*f1*t + cfg.phi));
        case 3 % Stepwise
            knot_t = linspace(0, t(end), numel(cfg.step_levels));
            u_raw = interp1(knot_t, cfg.step_levels, t, 'previous', 'extrap');
            u = conv(u_raw, ones(1,21)/21, 'same'); 
        case 4 % Triangular
            u = cfg.triangle_amplitude * sawtooth(2*pi*cfg.triangle_frequency*t, 0.5);
        case 5 % Concentric Loops
            turning_points = 0;
            A_vals = linspace(cfg.concentric_min_A, cfg.concentric_max_A, cfg.concentric_cycles);
            for A = A_vals, turning_points = [turning_points, A, -A, A]; end
            u = interpolate_points(turning_points, t);
        case 6 % FORC
            % Sequenza di inversioni
            u_r_values = linspace(0.9, -0.9, cfg.forc_steps);
            turning_points = [];
            if strcmp(cfg.forc_direction, 'top')
                for ur = u_r_values, turning_points = [turning_points, 1, ur, 1]; end
            else
                for ur = u_r_values, turning_points = [turning_points, -1, ur, -1]; end
            end
            u = interpolate_points(turning_points, t);
        case 7 % Dense Minor Loops
            u = cfg.A1 * sin(2*pi*f1*t) + cfg.dense_amps * sin(2*pi*cfg.dense_f_ratios*f1*t + cfg.phi);
    end
    
    u = u / (max(abs(u)) + 1e-9); 
    
    % Simulazione Preisach
    S = -ones(size(P));
    if cfg.type == 5, S(Alpha + Beta <= 0) = 1; end 
    M = zeros(size(t));
    da = alpha(2)-alpha(1);
    for k = 1:length(t)
        uk = u(k);
        S(Alpha <= uk) = 1;
        S(Beta >= uk) = -1;
        M(k) = sum(sum(P .* S)) * da^2;
    end
end

function u = interpolate_points(pts, t)
    dist = abs(diff(pts)); dist(dist==0) = 1e-6;
    tw = dist / sum(dist);
    kt = [0, cumsum(tw) * t(end)];
    u = zeros(size(t));
    for i = 1:length(pts)-1
        idx = (t >= kt(i)) & (t <= kt(i+1));
        if ~any(idx), continue; end
        seg_t = t(idx);
        phase = pi * (seg_t - kt(i)) / (kt(i+1) - kt(i));
        u(idx) = pts(i) + (pts(i+1) - pts(i)) * (1 - cos(phase)) / 2;
    end
end

function name = get_excitation_name(type)
    names = {'Multi-Sine', 'Damped Sine', 'Stepwise', 'Triangular', 'Concentric Loops', 'FORC', 'Dense Minor Loops'};
    if type >= 1 && type <= 7, name = names{type}; else, name = 'Sconosciuta'; end
end
