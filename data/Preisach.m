
% Modello di Preisach per ciclo di isteresi con eccitazione sinusoidale
% contenente un'armonica superiore

clear; close all; clc;

%% Parametri del modello
% Definizione della distribuzione di Preisach (funzione peso)
mu = @(alpha, beta) exp(-(alpha.^2 + beta.^2)) .* (alpha >= beta);

% Parametri temporali
T = 2;            % Periodo della fondamentale [s]
f1 = 1/T;         % Frequenza fondamentale [Hz]
f2 = 3*f1;        % Frequenza dell'armonica superiore [Hz]
t_final = 4*T;    % Tempo di simulazione
dt = T/1000;      % Passo temporale
t = 0:dt:t_final;

%% Segnale di ingresso (sinusoide con armonica)
A1 = 1.0;         % Ampiezza fondamentale
A2 = 0.5;         % Ampiezza armonica
phi = 0;          % Fase armonica

u = A1*sin(2*pi*f1*t) + A2*sin(6*pi*f2*t + phi);
u = u / max(abs(u)); % Normalizzazione

%% Inizializzazione modello di Preisach
% Griglia per il piano di Preisach
N_grid = 101;
alpha = linspace(-1, 1, N_grid);
beta = linspace(-1, 1, N_grid);
[Alpha, Beta] = meshgrid(alpha, beta);

% Calcolo della distribuzione di Preisach sulla griglia
P = mu(Alpha, Beta);
P(Alpha <= Beta) = 0; % Solo triangolo superiore

% Stato degli switch (1 = ON, -1 = OFF)
S = -ones(N_grid, N_grid); % Inizialmente tutti OFF

% Memoria degli estremi
history = struct('max', [], 'min', []);

%% Simulazione
M = zeros(size(t)); % Magnetizzazione/uscita

for k = 1:length(t)
    % Ingresso corrente
    u_current = u(k);
    
    % Aggiornamento stati degli switch
    S = update_switches(S, Alpha, Beta, u_current, history);
    
    % Calcolo dell'uscita (integrale sulla distribuzione)
    M(k) = sum(sum(P .* S)) * (alpha(2)-alpha(1)) * (beta(2)-beta(1));
    
    % Aggiornamento storia degli estremi
    history = update_history(history, u_current);
end

%% Risultati e visualizzazioni
figure('Position', [100, 100, 1200, 500]);

% 1. Ciclo di isteresi
subplot(1,3,1);
plot(u, M, 'b-', 'LineWidth', 1.5);
xlabel('Ingresso u(t) [normalizzato]', 'FontSize', 12);
ylabel('Uscita M(t)', 'FontSize', 12);
title('Ciclo di Isteresi Preisach', 'FontSize', 14);
grid on;

% 2. Segnali nel tempo
subplot(1,3,2);
plot(t, u, 'r-', 'LineWidth', 1.5); hold on;
plot(t, M, 'b-', 'LineWidth', 1.5);
xlabel('Time [s]', 'FontSize', 12);
ylabel('Ampiezza', 'FontSize', 12);
title('Ingresso e Uscita nel Tempo', 'FontSize', 14);
legend('Ingresso u(t)', 'Uscita M(t)', 'Location', 'best');
grid on;

% 3. Distribuzione di Preisach
subplot(1,3,3);
imagesc(alpha, beta, P);
xlabel('\alpha', 'FontSize', 12);
ylabel('\beta', 'FontSize', 12);
title('Distribuzione di Preisach \mu(\alpha,\beta)', 'FontSize', 14);
colorbar;
axis square;

%% Analisi spettrale
figure('Position', [100, 600, 800, 400]);

% FFT dell'ingresso
subplot(1,2,1);
[f_U, U_spectrum] = calculate_fft(u, dt);
stem(f_U, abs(U_spectrum), 'r', 'LineWidth', 1.5);
xlabel('Frequenza [Hz]', 'FontSize', 12);
ylabel('Ampiezza', 'FontSize', 12);
title('Spettro Ingresso', 'FontSize', 14);
xlim([0, 5*f1]);
grid on;

% FFT dell'uscita
subplot(1,2,2);
[f_M, M_spectrum] = calculate_fft(M, dt);
stem(f_M, abs(M_spectrum), 'b', 'LineWidth', 1.5);
xlabel('Frequenza [Hz]', 'FontSize', 12);
ylabel('Ampiezza', 'FontSize', 12);
title('Spettro Uscita', 'FontSize', 14);
xlim([0, 5*f1]);
grid on;

%% Informazioni diagnostiche
fprintf('=== MODELLO DI PREISACH ===\n');
fprintf('Frequenza fondamentale: %.2f Hz\n', f1);
fprintf('Frequenza armonica: %.2f Hz (%.1f x fondamentale)\n', f2, f2/f1);
fprintf('Ampiezza fondamentale: %.2f\n', A1);
fprintf('Ampiezza armonica: %.2f\n', A2);
fprintf('Rapporto armonica/fondamentale: %.2f\n', A2/A1);
fprintf('Tempo di simulazione: %.2f s\n', t_final);
fprintf('Numero punti griglia: %d x %d\n', N_grid, N_grid);
fprintf('Area ciclo isteresi: %.4f\n', abs(trapz(u, M)));


%% Funzioni ausiliarie
function S = update_switches(S, Alpha, Beta, u_current, history)
    % Aggiorna gli stati degli switch di Preisach
    [N, M] = size(S);
    
    % Determinazione dello stato in base alla regola di Preisach
    for i = 1:N
        for j = 1:M
            if Alpha(i,j) <= u_current
                S(i,j) = 1; % Switch ON
            elseif Beta(i,j) >= u_current
                S(i,j) = -1; % Switch OFF
            end
            % Altrimenti mantiene lo stato precedente
        end
    end
end

function history = update_history(history, u_current)
    % Aggiorna la storia degli estremi
    if isempty(history.max) || u_current > history.max(end)
        history.max(end+1) = u_current;
    end
    if isempty(history.min) || u_current < history.min(end)
        history.min(end+1) = u_current;
    end
end

function [f, spectrum] = calculate_fft(signal, dt)
    % Calcola la FFT di un segnale
    N = length(signal);
    Fs = 1/dt;
    spectrum = fft(signal)/N;
    spectrum = spectrum(1:floor(N/2)+1);
    spectrum(2:end-1) = 2*spectrum(2:end-1);
    f = Fs*(0:(N/2))/N;
end
