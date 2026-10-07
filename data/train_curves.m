% Script per testare e visualizzare manualmente le curve di isteresi
% generate con il modello di Preisach. Utile per sperimentare con i
% parametri e costruire un training set solido.

clear; close all; clc;

%% 1. DEFINIZIONE GRIGLIA DI PREISACH
% Funzione di densità di Preisach (distribuzione normale bivariata)
mu     = @(alpha, beta) exp(-(alpha.^2 + beta.^2)) .* (alpha >= beta);
N_grid = 101;
alpha  = linspace(-1, 1, N_grid);
beta   = linspace(-1, 1, N_grid);
[Alpha, Beta] = meshgrid(alpha, beta);
P = mu(Alpha, Beta);
P(Alpha <= Beta) = 0; % Sotto la diagonale la probabilità è nulla

%% 2. PARAMETRI DI TEMPO GENERALI
T       = 0.7;                % Periodo fondamentale [s]
f1      = 1 / T;              % Frequenza fondamentale [Hz]
t_final = 4 * T;              % Durata simulazione [s]
n_time_points = 2000;         % Risoluzione temporale
dt      = t_final / (n_time_points - 1);

%% 3. CONFIGURAZIONE ECCITAZIONE MANUALE
% parametri per "giocare" con le forme d'onda
curve_cfg = struct();

% Scegli il tipo di eccitazione: 
% 1 = Multi-Sine (Sinusoide + armoniche)
% 2 = Damped Sine (Sinusoide smorzata)
% 3 = Stepwise (Stepwise)
% 4 = Triangular (Triangular)
% 5 = Concentric Loops (Cicli Concentric Loops)
% 6 = FORC (Curve di prima inversione)
% 7 = Dense Minor Loops (Sinusoide con ricircoletti densi/armoniche)
curve_cfg.excitation_type = 1; 

% --- Parametri comuni (influiscono sulle ampiezze e forme d'onda di base) ---
curve_cfg.A1       = 0.8;      % Ampiezza della componente fondamentale
curve_cfg.A2       = 0.4;      % Ampiezza dell'armonica secondaria
curve_cfg.f2_ratio = 5;        % Moltiplicatore frequenza armonica (es. f2 = 5*f1)
curve_cfg.phi      = pi/3;     % Sfasamento dell'armonica [rad]

% --- Parametri specifici per Damped Sine (Tipo 2) ---
curve_cfg.tau      = 1.5 * T;  % Costante di tempo del Damped Sine esponenziale

% --- Parametri specifici per Stepwise (Tipo 3) ---
% Cambia i livelli per creare percorsi di isteresi complessi (valori tra -1 e 1)
curve_cfg.step_levels = [0.00, 0.60, -0.30, 0.85, -0.50, 0.95, -0.80, 0.40, -0.10, 0.00];

% --- Parametri specifici per Triangular (Tipo 4) ---
curve_cfg.triangle_amplitude = 0.8; % Ampiezza del segnale Triangular
curve_cfg.triangle_frequency = f1;   % Frequenza del segnale Triangular

% --- Parametri specifici per Concentric Loops (Tipo 5) ---
curve_cfg.concentric_cycles = 5;    % Numero di cicli
curve_cfg.concentric_min_A  = 0.5; % Ampiezza ciclo minimo
curve_cfg.concentric_max_A  = 1.0;  % Ampiezza ciclo massimo

% --- Parametri specifici per FORC / Prima inversione (Tipo 6) ---
curve_cfg.forc_steps = 8;           % Numero di curve di inversione (quanti ur generare)

% --- Parametri specifici per Ricircoletti densi (Tipo 7) ---
% Modula componenti armoniche per riempire lo spazio
% curve_cfg.dense_f_ratios = [2, 3, 5, 7, 13, 21]; % Multipli della frequenza base (f1)
% curve_cfg.dense_amps     = [0.25, 0.31, 0.22, 0.15, 0.1, 0.05]; % Ampiezze delle relative armoniche
curve_cfg.dense_f_ratios = [28]; 
curve_cfg.dense_amps     = [0.3];

%% 4. SIMULAZIONE DEL MODELLO DI PREISACH
fprintf('Avvio simulazione modello Preisach...\n');
[u, M, t] = simulate_preisach_manual(curve_cfg, f1, t_final, dt, Alpha, Beta, P, alpha);
fprintf('Simulazione completata.\n');


%% 5. VISUALIZZAZIONE RISULTATI
fig = figure('Name', 'Manual Preisach Analysis', 'Position', [100, 100, 1400, 450]);

% 5.1 Plot Eccitazione u(t)
subplot(1, 3, 1);
plot(t, u, 'b', 'LineWidth', 1.5);
xlabel('Time [s]'); 
ylabel('Applied Field u(t)');
title(sprintf('Input Signal (Type: %d)', curve_cfg.excitation_type));
grid on; 
ylim([-1.1 1.1]);

% 5.2 Plot Magnetizzazione M(t)
subplot(1, 3, 2);
plot(t, M, 'r', 'LineWidth', 1.5);
xlabel('Time [s]'); 
ylabel('Magnetization M(t)');
title('Preisach Model Output');
grid on;

% 5.3 Plot Ciclo di Isteresi M(u)
subplot(1, 3, 3);
plot(u, M, 'k-', 'LineWidth', 1.5);
xlabel('u (Applied Field)'); 
ylabel('M (Magnetization)');
title('Hysteresis Loop');
grid on;

% Aggiungo un puntino rosso per indicare la fine del ciclo di isteresi
hold on;
plot(u(end), M(end), 'ro', 'MarkerSize', 6, 'MarkerFaceColor', 'r');
hold off;

%% 6. ARTICLE-READY HYSTERESIS LOOP PLOT
type_names = {'Multi-Sine', 'Damped Sine', 'Stepwise', 'Triangular', ...
              'Concentric Loops', 'FORC', 'Dense Minor Loops'};
current_type = type_names{curve_cfg.excitation_type};

fig_article = figure('Name', sprintf('Hysteresis Loop - %s', current_type), 'Position', [150, 150, 700, 550]);
plot(u, M, 'k-', 'LineWidth', 2.5, 'DisplayName', 'Preisach Model');
xlabel('Applied Field \it H \rm [A/m]', 'FontSize', 12); 
ylabel('Magnetization \it M \rm [A/m]', 'FontSize', 12);
title(sprintf('Hysteresis Loop - %s', current_type), 'FontSize', 14);
grid on;
set(gca, 'FontSize', 11, 'LineWidth', 1.0);
legend('Location', 'best', 'FontSize', 11);


%% ============================================================
% FUNZIONI LOCALI DI SUPPORTO
% ============================================================

function [u, M, t] = simulate_preisach_manual(curve_cfg, f1, t_final, dt, Alpha, Beta, P, alpha)
    t = 0:dt:t_final;
    f2 = curve_cfg.f2_ratio * f1;
    
    % --- 1. Generazione del segnale di ingresso u(t) ---
    switch curve_cfg.excitation_type
        case 1
            u = curve_cfg.A1 * sin(2*pi*f1*t) + ...
                curve_cfg.A2 * sin(2*pi*f2*t + curve_cfg.phi);
                
        case 2
            envelope = exp(-t / curve_cfg.tau);
            base_signal = curve_cfg.A1 * sin(2*pi*f1*t) + ...
                curve_cfg.A2 * sin(2*pi*f2*t + curve_cfg.phi);
            u = envelope .* base_signal;
            
        case 3
            knot_t = linspace(0, t(end), numel(curve_cfg.step_levels));
            u_steps = interp1(knot_t, curve_cfg.step_levels, t, 'previous', 'extrap');
            
            % Smoothing locale per evitare gradienti infiniti
            smooth_span = max(5, 2 * floor(0.02 * numel(t) / 2) + 1);
            kernel = ones(1, smooth_span) / smooth_span;
            u = conv(u_steps, kernel, 'same');
            
            % Aggiunta di piccolo ripple per dinamica fluida
            ripple = 0.05 * (curve_cfg.A1 * sin(2*pi*f1*t + curve_cfg.phi) + ...
                0.5 * curve_cfg.A2 * sin(2*pi*f2*t));
            u = u + ripple;
            
        case 4
            u = curve_cfg.triangle_amplitude * sawtooth(2*pi*curve_cfg.triangle_frequency*t, 0.5);
            
        case 5
            % Cicli Concentric Loops espliciti e perfettamente chiusi
            N_cycles = curve_cfg.concentric_cycles;
            A_concentric = linspace(curve_cfg.concentric_min_A, curve_cfg.concentric_max_A, N_cycles);
            
            % 1. Costruiamo i punti di inversione (turning points)
            % Per ogni ampiezza A: vai a A, scendi a -A, torna ad A (chiude il ciclo)
            turning_points = 0;
            for i = 1:N_cycles
                A = A_concentric(i);
                turning_points = [turning_points, A, -A, A];
            end
            
            % 2. Mappiamo il tempo in modo proporzionale alla distanza 
            % per mantenere pendenze omogenee
            distances = abs(diff(turning_points));
            distances(distances == 0) = 1e-6; % Evita divisioni per zero
            time_weights = distances / sum(distances);
            knot_t = [0, cumsum(time_weights) * t(end)];
            
            u = zeros(size(t));
            for i = 1:length(turning_points)-1
                if i == length(turning_points)-1
                    idx = (t >= knot_t(i));
                else
                    idx = (t >= knot_t(i)) & (t < knot_t(i+1));
                end
                
                t_seg = t(idx);
                dt_seg = knot_t(i+1) - knot_t(i);
                phase = pi * (t_seg - knot_t(i)) / dt_seg;
                % Interpolazione morbida con emicoseno
                u(idx) = turning_points(i) + (turning_points(i+1) - turning_points(i)) * (1 - cos(phase)) / 2;
            end
            
        case 6
            % FORC (First Order Reversal Curves): 
            % Satura a +1, scende a un valore u_r prestabilito (inversione), e risale a +1.
            u = zeros(size(t));
            N_steps = curve_cfg.forc_steps;
            u_r_values = linspace(0.85, -0.95, N_steps); % Sequenza di campi di inversione u_r
            t_segment = t_final / N_steps;
            for k = 1:length(t)
                seg_idx = floor(t(k) / t_segment) + 1;
                seg_idx = min(seg_idx, N_steps);
                tau = (t(k) - (seg_idx-1)*t_segment) / t_segment; % Normalizzato [0, 1] nel micro-segmento
                u_r = u_r_values(seg_idx);
                % Interpolazione "smooth" con coseno per evitare picchi discontinui nella derivata
                u(k) = (1 + u_r)/2 + (1 - u_r)/2 * cos(2*pi*tau);
            end
            
        case 7
            % Ricircoletti densi: Fondamentale + super-armoniche per minor loops
            u = curve_cfg.A1 * sin(2*pi*f1*t);
            for i = 1:length(curve_cfg.dense_f_ratios)
                f_n = curve_cfg.dense_f_ratios(i) * f1;
                A_n = curve_cfg.dense_amps(i);
                u = u + A_n * sin(2*pi*f_n*t);
            end
            
        otherwise
            error('Tipo di eccitazione %d non valido.', curve_cfg.excitation_type);
    end
    
    % Normalizzazione del segnale per metterlo nel range [-1, 1]
    max_abs = max(abs(u));
    if max_abs > 0
        u = u / max_abs;
    end
    u = max(min(u, 1), -1);

    % --- 2. Modello di Preisach ---
    N_grid = size(Alpha, 1);
    S      = -ones(N_grid, N_grid); % Stato iniziale magnetizzazione (tutti negativi)
    M      = zeros(size(t));
    da     = alpha(2) - alpha(1);

    % Fondamentale: per avere cicli Concentric Loops perfettamente simmetrici 
    % bisogna partire dallo stato smagnetizzato neutro fin dall'inizio.
    if curve_cfg.excitation_type == 5
         S(Alpha + Beta <= 0) = 1; 
    end

    for k = 1:length(t)
        % NESSUN RESET MANUALE: la fisica del modello cancella le storie minori
        
        u_k = u(k);
        % Aggiornamento operatori di relay (operatore bistabile)
        S(Alpha <= u_k) = 1;
        S(Beta >= u_k)  = -1;
        
        % Calcolo integrale di Preisach approcciato con la somma
        M(k) = sum(sum(P .* S)) * da^2;
    end
end
