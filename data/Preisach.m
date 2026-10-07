% Preisach Hysteresis Model Simulation
% Simulates magnetic hysteresis loops under sinusoidal excitation with harmonic distortion.

clear; close all; clc;

%% Model Parameters
% Definition of the Preisach weight distribution (bivariate Gaussian)
mu = @(alpha, beta) exp(-(alpha.^2 + beta.^2)) .* (alpha >= beta);

% Temporal parameters
T = 2;            % Fundamental period [s]
f1 = 1/T;         % Fundamental frequency [Hz]
f2 = 3*f1;        % Higher harmonic frequency [Hz]
t_final = 4*T;    % Total simulation time [s]
dt = T/1000;      % Time step [s]
t = 0:dt:t_final;

%% Applied Excitation Field (Fundamental + Harmonic)
A1 = 1.0;         % Fundamental amplitude
A2 = 0.5;         % Harmonic amplitude
phi = 0;          % Harmonic phase [rad]

u = A1*sin(2*pi*f1*t) + A2*sin(6*pi*f2*t + phi);
u = u / max(abs(u)); % Dimensionless normalization to [-1, 1]

%% Preisach Grid Initialization
% Discretization grid on the (alpha, beta) plane
N_grid = 101;
alpha = linspace(-1, 1, N_grid);
beta = linspace(-1, 1, N_grid);
[Alpha, Beta] = meshgrid(alpha, beta);

% Evaluate weight distribution on grid
P = mu(Alpha, Beta);
P(Alpha <= Beta) = 0; % Upper triangle only (alpha >= beta)

% Elementary dipoles / switch states (1 = ON, -1 = OFF)
S = -ones(N_grid, N_grid); % Initialized in negative saturation (-1)

% Reversal history tracking
history = struct('max', [], 'min', []);

%% Simulation Loop
M = zeros(size(t)); % Magnetization output

for k = 1:length(t)
    u_current = u(k);
    
    % Update dipole switch states
    S = update_switches(S, Alpha, Beta, u_current, history);
    
    % Compute magnetization integral
    M(k) = sum(sum(P .* S)) * (alpha(2)-alpha(1)) * (beta(2)-beta(1));
    
    % Update extrema history
    history = update_history(history, u_current);
end

%% Visualization
figure('Position', [100, 100, 1200, 500]);

% 1. Hysteresis Loop
subplot(1,3,1);
plot(u, M, 'b-', 'LineWidth', 1.5);
xlabel('Applied Field u(t) [-]', 'FontSize', 12);
ylabel('Magnetization M(t) [-]', 'FontSize', 12);
title('Preisach Hysteresis Loop', 'FontSize', 14);
grid on;

% 2. Time-Domain Signals
subplot(1,3,2);
plot(t, u, 'r-', 'LineWidth', 1.5); hold on;
plot(t, M, 'b-', 'LineWidth', 1.5);
xlabel('Time [s]', 'FontSize', 12);
ylabel('Normalized Amplitude [-]', 'FontSize', 12);
title('Excitation and Response in Time', 'FontSize', 14);
legend('Applied Field u(t)', 'Magnetization M(t)', 'Location', 'best');
grid on;

% 3. Preisach Weight Distribution
subplot(1,3,3);
imagesc(alpha, beta, P);
xlabel('lpha', 'FontSize', 12);
ylabel('eta', 'FontSize', 12);
title('Preisach Distribution \mu(lpha,eta)', 'FontSize', 14);
colorbar;
axis square;

%% Spectral Analysis
figure('Position', [100, 600, 800, 400]);

% FFT of Input
subplot(1,2,1);
[f_U, U_spectrum] = calculate_fft(u, dt);
stem(f_U, abs(U_spectrum), 'r', 'LineWidth', 1.5);
xlabel('Frequency [Hz]', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
title('Input Spectrum', 'FontSize', 14);
xlim([0, 5*f1]);
grid on;

% FFT of Output
subplot(1,2,2);
[f_M, M_spectrum] = calculate_fft(M, dt);
stem(f_M, abs(M_spectrum), 'b', 'LineWidth', 1.5);
xlabel('Frequency [Hz]', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
title('Output Spectrum', 'FontSize', 14);
xlim([0, 5*f1]);
grid on;

%% Diagnostic Printout
fprintf('=== PREISACH MODEL SIMULATION ===\n');
fprintf('Fundamental frequency: %.2f Hz\n', f1);
fprintf('Harmonic frequency:    %.2f Hz (%.1f x fundamental)\n', f2, f2/f1);
fprintf('Fundamental amplitude: %.2f\n', A1);
fprintf('Harmonic amplitude:    %.2f\n', A2);
fprintf('Harmonic/Fundamental:  %.2f\n', A2/A1);
fprintf('Simulation duration:   %.2f s\n', t_final);
fprintf('Grid discretization:   %d x %d operators\n', N_grid, N_grid);
fprintf('Loop area (energy):    %.4f\n', abs(trapz(u, M)));

%% Helper Functions
function S = update_switches(S, Alpha, Beta, u_current, history)
    % Update Preisach switch states according to switching thresholds
    [N, M_dim] = size(S);
    for i = 1:N
        for j = 1:M_dim
            if Alpha(i,j) <= u_current
                S(i,j) = 1;  % Switch ON
            elseif Beta(i,j) >= u_current
                S(i,j) = -1; % Switch OFF
            end
        end
    end
end

function history = update_history(history, u_current)
    % Update extrema history
    if isempty(history.max) || u_current > history.max(end)
        history.max(end+1) = u_current;
    end
    if isempty(history.min) || u_current < history.min(end)
        history.min(end+1) = u_current;
    end
end

function [f, spectrum] = calculate_fft(signal, dt)
    % Compute one-sided FFT spectrum
    N = length(signal);
    Fs = 1/dt;
    spectrum = fft(signal)/N;
    spectrum = spectrum(1:floor(N/2)+1);
    spectrum(2:end-1) = 2*spectrum(2:end-1);
    f = Fs*(0:(N/2))/N;
end
