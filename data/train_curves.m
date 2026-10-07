% Interactive Preisach Hysteresis Curve Generator
% Allows interactive generation and inspection of individual hysteresis loops.

clear; close all; clc;

%% 1. Preisach Discretization Grid
mu     = @(alpha, beta) exp(-(alpha.^2 + beta.^2)) .* (alpha >= beta);
N_grid = 101;
alpha  = linspace(-1, 1, N_grid);
beta   = linspace(-1, 1, N_grid);
[Alpha, Beta] = meshgrid(alpha, beta);
P = mu(Alpha, Beta);
P(Alpha <= Beta) = 0;

%% 2. Global Temporal Parameters
T       = 0.7;                % Fundamental period [s]
f1      = 1 / T;              % Fundamental frequency [Hz]
t_final = 4 * T;              % Simulation duration [s]
n_time_points = 2000;         % Time resolution
dt      = t_final / (n_time_points - 1);

%% 3. Excitation Waveform Configuration
curve_cfg = struct();

% Excitation Type:
% 1 = Multi-Sine (Fundamental + Harmonics)
% 2 = Damped Sine (Exponentially Decaying Sinusoid)
% 3 = Stepwise (Discontinuous Staircase)
% 4 = Triangular (Constant Rate of Change)
% 5 = Concentric Loops (Nested Inner Minor Loops)
% 6 = FORC (First-Order Reversal Curves)
% 7 = Dense Minor Loops (High-Frequency Reversals)
curve_cfg.excitation_type = 1; 

% --- Common Parameters ---
curve_cfg.A1       = 0.8;      % Fundamental amplitude
curve_cfg.A2       = 0.4;      % Harmonic amplitude
curve_cfg.f2_ratio = 5;        % Harmonic ratio (f2 = 5*f1)
curve_cfg.phi      = pi/3;     % Harmonic phase offset [rad]

% --- Parameters for Damped Sine (Type 2) ---
curve_cfg.tau      = 1.5 * T;  % Exponential decay time constant

% --- Parameters for Stepwise (Type 3) ---
curve_cfg.step_levels = [0.00, 0.60, -0.30, 0.85, -0.50, 0.95, -0.80, 0.40, -0.10, 0.00];

% --- Parameters for Triangular (Type 4) ---
curve_cfg.triangle_amplitude = 0.8;
curve_cfg.triangle_frequency = f1;

% --- Parameters for Concentric Loops (Type 5) ---
curve_cfg.concentric_cycles = 5;
curve_cfg.concentric_min_A  = 0.5;
curve_cfg.concentric_max_A  = 1.0;

% --- Parameters for FORC (Type 6) ---
curve_cfg.forc_steps = 8;

% --- Parameters for Dense Minor Loops (Type 7) ---
curve_cfg.dense_f_ratios = [28]; 
curve_cfg.dense_amps     = [0.3];

%% 4. Preisach Model Simulation
t = 0:dt:t_final;

switch curve_cfg.excitation_type
    case 1 % Multi-Sine
        u = curve_cfg.A1 * sin(2*pi*f1*t) + curve_cfg.A2 * sin(2*pi*curve_cfg.f2_ratio*f1*t + curve_cfg.phi);
        curve_name = 'Multi-Sine';
        
    case 2 % Damped Sine
        u = exp(-t/curve_cfg.tau) .* (curve_cfg.A1 * sin(2*pi*f1*t) + curve_cfg.A2 * sin(2*pi*curve_cfg.f2_ratio*f1*t + curve_cfg.phi));
        curve_name = 'Damped Sine';
        
    case 3 % Stepwise
        knot_t = linspace(0, t_final, numel(curve_cfg.step_levels));
        u_raw = interp1(knot_t, curve_cfg.step_levels, t, 'previous', 'extrap');
        u = conv(u_raw, ones(1,21)/21, 'same');
        curve_name = 'Stepwise';
        
    case 4 % Triangular
        u = curve_cfg.triangle_amplitude * sawtooth(2*pi*curve_cfg.triangle_frequency*t, 0.5);
        curve_name = 'Triangular';
        
    case 5 % Concentric Loops
        turning_points = 0;
        A_vals = linspace(curve_cfg.concentric_min_A, curve_cfg.concentric_max_A, curve_cfg.concentric_cycles);
        for A = A_vals
            turning_points = [turning_points, A, -A, A];
        end
        u = interpolate_turning_points(turning_points, t);
        curve_name = 'Concentric Loops';
        
    case 6 % FORC
        u_r_values = linspace(0.9, -0.9, curve_cfg.forc_steps);
        turning_points = [];
        for ur = u_r_values
            turning_points = [turning_points, 1, ur, 1];
        end
        u = interpolate_turning_points(turning_points, t);
        curve_name = 'FORC';
        
    case 7 % Dense Minor Loops
        u = curve_cfg.A1 * sin(2*pi*f1*t);
        for h = 1:numel(curve_cfg.dense_f_ratios)
            u = u + curve_cfg.dense_amps(h) * sin(2*pi*curve_cfg.dense_f_ratios(h)*f1*t + curve_cfg.phi);
        end
        curve_name = 'Dense Minor Loops';
end

u = u / (max(abs(u)) + 1e-9);

% Dipole state initialization
S = -ones(N_grid, N_grid);
if curve_cfg.excitation_type == 5
    S(Alpha + Beta <= 0) = 1;
end

M = zeros(size(t));
da = alpha(2) - alpha(1);

for k = 1:length(t)
    uk = u(k);
    S(Alpha <= uk) = 1;
    S(Beta >= uk)  = -1;
    M(k) = sum(sum(P .* S)) * (da^2);
end

du_dt = gradient(u, dt);
du_dt = du_dt / (max(abs(du_dt)) + 1e-9);

%% 5. Visualization
figure('Position', [100, 100, 1200, 500], 'Name', ['Inspection: ', curve_name]);

subplot(1, 2, 1);
plot(u, M, 'b-', 'LineWidth', 1.8);
xlabel('Applied Field u(t) [-]', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Magnetization M(t) [-]', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Hysteresis Loop: %s', curve_name), 'FontSize', 14, 'FontWeight', 'bold');
grid on;
xlim([-1.05, 1.05]); ylim([-1.05, 1.05]);

subplot(1, 2, 2);
plot(t, u, 'r-', 'LineWidth', 1.5, 'DisplayName', 'u(t)'); hold on;
plot(t, M, 'b-', 'LineWidth', 1.5, 'DisplayName', 'M(t)');
xlabel('Time [s]', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Amplitude [-]', 'FontSize', 12, 'FontWeight', 'bold');
title('Time-Domain Signals', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;

fprintf('Generated curve: %s | Energy area: %.4f\n', curve_name, abs(trapz(u, M)));

%% Helper Function
function u = interpolate_turning_points(pts, t)
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
