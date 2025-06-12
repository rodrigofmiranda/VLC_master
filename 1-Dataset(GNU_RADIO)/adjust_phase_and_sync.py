#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
import os
import concurrent.futures
from datetime import datetime
import json
matplotlib.use('Agg')

# Configuração de formato de exportação: 'pdf' ou 'svg'
export_format = 'svg'

# Diretório de saída para plots
save_dir = 'plots'
os.makedirs(save_dir, exist_ok=True)

# Executor para salvar figuras em paralelo
executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

def save_fig_in_background(fig, basename, **kwargs):
    # Monta caminho com extensão configurada
    filepath = os.path.join(save_dir, f"{basename}.{export_format}")
    executor.submit(fig.savefig, filepath, **kwargs)
    plt.close(fig)

skip = 0.1 # % of the cut of the dataset
ws = 0.01 # % of windows dataset to calculate the EVM 

# Specify the file paths for the transmitted (sent) and received I/Q data.
enviado_path = 'sent'  # Raw binary files without an extension.
recebido_path = 'received'

# Read I/Q data from binary files as 32-bit floats.
with open(enviado_path, 'rb') as f:
    x_float32 = np.fromfile(f, dtype=np.float32)
with open(recebido_path, 'rb') as f:
    y_float32 = np.fromfile(f, dtype=np.float32)

print(f"length of y: {len(y_float32)}")
print(f"length of x: {len(x_float32)}")

# Ensure 'y_float32' is trimmed to match the length of 'x_float32'.

if len(y_float32) > len(x_float32):
    jump = round(skip * len(x_float32))
    # → força 'jump' a ser múltiplo de 2
    jump -= jump % 2
    y_float32 = y_float32[jump:len(x_float32)]
    x_float32 = x_float32[jump:len(x_float32)]
else:
    jump = round(skip * len(y_float32))
    jump -= jump % 2
    x_float32 = x_float32[jump:len(y_float32)]
    y_float32 = y_float32[jump:len(y_float32)]

# (pós-corte) se ainda restar tamanho ímpar, descarta 1 elemento em cada vetor
min_len = min(len(x_float32), len(y_float32))
if min_len % 2:           # se for ímpar
    min_len -= 1
x_float32 = x_float32[:min_len]
y_float32 = y_float32[:min_len]

def extract_parameters_and_combine(script_path):
    """
    Extracts essential parameters from the GNU Radio generated Python script,
    combines device address and stream channel information,
    and returns a dict of extracted parameters.
    """
    regexes = {
        'samples': r'self\.samples = samples = ([\d\.e\+\-]+)',
        'samp_rate': r'self\.samp_rate = samp_rate = ([\d\.e\+\-]+)',
        'sps': r'sps = sps = ([\d\.e\+\-]+)',
        'nfilts': r'self\.nfilts = nfilts = ([\d\.e\+\-]+)',
        'tuning': r'self\.tuning = tuning = ([\d\.e\+\-]+)',
        'gain': r'self\.gain = gain = ([\d\.e\+\-]+)',
        'frequency': r'self\.frequency = frequency = ([\d\.e\+\-]+)',
        'rf_gain': r'self\.rf_gain = rf_gain = ([\d\.e\+\-]+)',
        'phase_bw': r'self\.phase_bw = phase_bw = ([\d\.e\+\-]+)',
        'excess_bw': r'self\.excess_bw = excess_bw = ([\d\.e\+\-]+)',
        'arity': r'self\.arity = arity = (\d+)',
        'rrc_taps': r'self\.rrc_taps = rrc_taps = (firdes\.root_raised_cosine\([^\)]+\))',
        'clock_source': r'set_clock_source\(\s*["\']([^"\']+)["\']',
        'antenna': r'set_antenna\(\s*["\']([^"\']+)["\']',
        'bandwidth': r'set_bandwidth\(\s*([^,]+),',
        'device_address_source': r'uhd\.usrp_source\(\s*"?addr0=[^\"]+"?',
        'stream_channel_source': r'uhd\.usrp_source\(.+?channels=\[([0-9]+)\]',
        'device_address_sink': r'uhd\.usrp_sink\(\s*"?addr0=[^\"]+"?',
        'stream_channel_sink': r'uhd\.usrp_sink\(.+?channels=\[([0-9]+)\]'
    }
    parameters = {}
    with open(script_path, 'r') as file:
        content = file.read()
        for key, pattern in regexes.items():
            match = re.search(pattern, content)
            if match:
                parameters[key] = match.group(1)

    # Combine device address and channel for source and sink
    if 'device_address_source' in parameters and 'stream_channel_source' in parameters:
        parameters['source_address'] = (
            f"{parameters.pop('device_address_source')} (Channel: {parameters.pop('stream_channel_source')})"
        )
    if 'device_address_sink' in parameters and 'stream_channel_sink' in parameters:
        parameters['sink_address'] = (
            f"{parameters.pop('device_address_sink')} (Channel: {parameters.pop('stream_channel_sink')})"
        )

    # Add skip and window size
    parameters['skip'] = skip
    parameters['window_size'] = ws

    # Calculate duration if samples and samp_rate are available
    if 'samples' in parameters and 'samp_rate' in parameters:
        try:
            num = float(parameters['samples'])
            rate = float(parameters['samp_rate'])
            parameters['duration_s'] = round(num / rate, 6)
        except ValueError:
            pass

    return parameters

from datetime import datetime

def print_and_save_with_combined_params(script_path, message, file_path='output.txt'):
    """
    Extrai parâmetros, adiciona timestamp e formata o log com caixas e colunas.
    """
    params = extract_parameters_and_combine(script_path)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Preparar linhas de parâmetro com largura fixa
    key_width = max(len(k) for k in params) + 2
    lines = []
    for k, v in sorted(params.items()):
        lines.append(f"│ {k.ljust(key_width)}│ {v}")

    border = "┌" + "─"*(key_width+2) + "┬" + "─"*40 + "┐"
    footer = "└" + "─"*(key_width+2) + "┴" + "─"*40 + "┘"

    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(f"=== {timestamp} | {message} ===\n")
        f.write(border + "\n")
        f.write(f"│ {'Parâmetro'.ljust(key_width)}│ Valor{' '*36}│\n")
        f.write(border.replace('┌','├').replace('┐','┤') + "\n")
        for line in lines:
            f.write(line + "\n")
        f.write(footer + "\n\n")

script_path = "channel_dataset.py"
initial_message = ""
print_and_save_with_combined_params(script_path, initial_message)

# This function serves dual purposes: it prints messages to the console and saves them to a specified file.
def print_and_save(message, file_path='output.txt'):
    print(message)  # Display the message on the console.
    with open(file_path, 'a') as f:  # Open the file in append mode or create it if it doesn't exist.
        f.write(message + '\n')  # Write the message to the file and move to a new line.

print(f"length of y (trimmed): {len(y_float32)}")
print(f"length of x (trimmed): {len(x_float32)}")

# Convert pairs of floats to complex samples for both transmitted and received signals.
IQ_x_complex = x_float32[::2] + 1j * x_float32[1::2]  # Transmitted I/Q samples.
IQ_y_complex = y_float32[::2] + 1j * y_float32[1::2]  # Received I/Q samples

def synchronize_and_normalize(transmitted_signal, received_signal, print_delay=True):
    """
    Synchronizes and normalizes the received signal with respect to the transmitted signal.
    It corrects the time delay and adjusts for phase and amplitude differences.
    
    Parameters:
    - transmitted_signal: The transmitted I/Q samples.
    - received_signal: The received I/Q samples.
    - print_delay: Flag to print detected and corrected delays.
    
    Returns:
    - The normalized received signal.
    """
    # Perform FFT on both signals for frequency domain analysis.
    fft_transmitted = np.fft.fft(transmitted_signal)
    fft_received = np.fft.fft(received_signal)

    # Calculate cross-correlation using IFFT to find the delay.
    cross_correlation = np.fft.ifft(fft_received * np.conj(fft_transmitted))
    delay = np.argmax(np.abs(cross_correlation)) # Find the index of maximum correlation which represents the delay.
    
    if delay > len(transmitted_signal) // 2:
        corrected_delay = delay - len(transmitted_signal)
    else:
        corrected_delay = delay
	
    # Print the detected delay before any correction.
    if print_delay:
        print_and_save(f"Detected delay before correction: {delay}")

    # Correct the delay if the detected delay exceeds half the length of the signal, indicating a wrap-around.

        
    # Align the received signal by rolling it back by the detected delay.
    received_signal_aligned = np.roll(received_signal, -corrected_delay)

    # Recalculate cross-correlation after alignment to verify correction. Ideally, the new delay should be zero.
    fft_received_aligned = np.fft.fft(received_signal_aligned)
    new_cross_correlation = np.fft.ifft(fft_received_aligned * np.conj(fft_transmitted))
    new_delay = np.argmax(np.abs(new_cross_correlation))
    
    # Ensures printing the recalculated delay accurately reflects the adjustment.
    if print_delay:
        print_and_save(f"Delay after correction: {new_delay}")

    # Adjust for phase differences between the transmitted and aligned received signals.
    phase_difference = np.angle(np.fft.fft(received_signal_aligned) / fft_transmitted)
    average_phase_difference = np.angle(np.mean(np.exp(1j * phase_difference)))
   
    # Apply the phase correction to the aligned received signal.
    received_signal_aligned_corrected = received_signal_aligned * np.exp(-1j * average_phase_difference)
    
    # Normalize the amplitude of the corrected received signal to match the transmitted signal.
    normalization_factor = np.sqrt(np.mean(np.abs(transmitted_signal)**2) / np.mean(np.abs(received_signal_aligned_corrected)**2))
    #normalization_factor=1
    IQ_y_complex_normalized = received_signal_aligned_corrected * normalization_factor

    return IQ_y_complex_normalized
# Process signals and print results.
IQ_y_complex_normalized = synchronize_and_normalize(IQ_x_complex, IQ_y_complex)   

print("length IQ_x_complex:",len(IQ_x_complex))
print("length IQ_y_complex:",len(IQ_y_complex))
print("length IQ_y_complex_normalized_full_sync:",len(IQ_y_complex_normalized))

# Function to estimate noise power.
def estimate_noise_power(transmitted_signal, received_signal):
    """
    Estimates the noise power based on the difference between the transmitted and received signals.
    
    Parameters:
    - transmitted_signal: The transmitted I/Q samples.
    - received_signal: The normalized received I/Q samples.
    
    Returns:
    - The estimated noise power.
    """
    noise_signal = received_signal - transmitted_signal  # Noise signal.
    noise_power = np.mean(np.abs(noise_signal) ** 2)  # Noise power calculation.
    return noise_power

def calculate_evm(transmitted_signal, received_signal):
    # Calcular o vetor de erro
    error_vector = received_signal - transmitted_signal
    # Calcular a magnitude do vetor de erro
    error_magnitude = np.abs(error_vector) 
    # Calcular a potência média dos símbolos transmitidos
    average_power = np.mean(np.abs(transmitted_signal)**2)    
    # Calcular EVM
    EVM = np.sqrt(np.mean(error_magnitude**2) / average_power)   
    # Converter EVM para porcentagem
    EVM_percentage = EVM * 100    
    # Convertendo EVM para dB
    EVM_dB = 20 * np.log10(EVM)

    return EVM_percentage, EVM_dB
    
def calculate_snr(transmitted_signal, received_signal):
    signal_power = np.mean(np.abs(transmitted_signal) ** 2)
    noise_power = np.mean(np.abs(received_signal - transmitted_signal) ** 2)
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db

# Calculate EVM
evm_percentage, evm_dB = calculate_evm(IQ_x_complex, IQ_y_complex_normalized)
print_and_save(f"EVM: {evm_percentage:.2f}%")
print_and_save(f"EVM (dB): {evm_dB:.2f} dB")

# Calculate noise power and SNR.
noise_power = estimate_noise_power(IQ_x_complex, IQ_y_complex_normalized)
print_and_save(f"Noise power: {noise_power}")

# Calculate SNR
SNR_dB = calculate_snr(IQ_x_complex, IQ_y_complex_normalized)
print_and_save(f"SNR: {SNR_dB} dB")

# Calcular a FFT
fft_resultado = np.fft.fft(IQ_y_complex_normalized)
fft_resultado2 = np.fft.fft(IQ_x_complex)

parameters = extract_parameters_and_combine(script_path)
samp_rate = float(parameters.get('samp_rate', 1))

# Calcular frequências correspondentes
freqs = np.fft.fftfreq(len(IQ_y_complex_normalized), 1 / samp_rate) 

# Função para calcular EVM em janelas, excluindo a última janela
def calculate_evm_windows(reference_signal, test_signal, window_size):
    num_windows = len(reference_signal) // window_size
    evm_percentages = []
    for i in range(num_windows - 1):
        start = i * window_size
        end   = start + window_size
        ref_win  = reference_signal[start:end]
        test_win = test_signal[start:end]

        # ── nova normalização local ──
        gain = np.sqrt(np.mean(np.abs(ref_win)**2) / np.mean(np.abs(test_win)**2))
        test_win = test_win * gain
        # ── ------------------------ ──

        evm_pct, _ = calculate_evm(ref_win, test_win)
        evm_percentages.append(evm_pct)
    return evm_percentages

# Definir o tamanho da janela (modificável)
window_size = int(ws * len(IQ_x_complex))  # ws% do tamanho do dataset, modifique conforme necessário

# Calcular EVM em janelas
evm_windows = calculate_evm_windows(IQ_x_complex, IQ_y_complex_normalized, window_size)

# Calcular EVM total para o dataset inteiro
EVM_percentage_total, EVM_dB_total = calculate_evm(IQ_x_complex, IQ_y_complex_normalized)

# Calcular a variação delta do EVM em porcentagem e dB
delta_evm_percentage = max(evm_windows) - min(evm_windows)
delta_evm_dB = 20 * np.log10(delta_evm_percentage / 100)

# depois de obter delta_evm_percentage
eps = 1e-12                      # evita log10(0)
delta_evm_linear = (delta_evm_percentage / 100) + eps
delta_evm_dB = 20 * np.log10(delta_evm_linear)

# Identifying the midpoint and selecting 100 middle samples.
mid_point = len(IQ_x_complex) // 2
start = mid_point - 50
end = mid_point + 50
IQ_x_complex_mid_samples = IQ_x_complex[start:end]
IQ_y_complex_normalized_mid_samples = IQ_y_complex_normalized[start:end]

# Função original para plotar EVM em janelas
def plot_evm_windows_with_total(evm_percentages, total_evm, delta_evm_percentage, delta_evm_dB, window_size, title_suffix, save_dir):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.gca()
    ax.plot(evm_percentages, marker='o', linestyle='-', label='EVM in Windows')
    ax.axhline(y=total_evm, color='r', linestyle='--', label='Total EVM')
    ax.set_xlabel('Window Index')
    ax.set_ylabel('EVM (%)')
    ax.set_title(f'EVM over Windows (Window Size: {window_size}) - {title_suffix}')
    ax.legend(loc='upper right')
    ax.grid(True)
    fig.text(0.15, 0.85,
             f'Delta EVM: {delta_evm_percentage:.2f}%, {delta_evm_dB:.2f} dB \nTotal EVM: {total_evm:.2f}%',
             fontsize=10, bbox={"facecolor":"orange","alpha":0.5,"pad":5})
    fig.tight_layout()
    save_fig_in_background(fig, f'evm_windows_{title_suffix}', dpi=300)

def plot_constellations(x, y, y_norm, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title in zip(axes,
                               [x, y, y_norm],
                               ['Sinal enviado', 'Sinal recebido original', 'Sinal recebido normalizado']):
        ax.scatter(data.real, data.imag, marker='.', s=0.01, rasterized=True)
        ax.set_title(title)
        ax.set_xlabel('Parte real')
        ax.set_ylabel('Parte imaginária')
        ax.set_aspect('equal')
        ax.grid(True)
    fig.tight_layout()
    save_fig_in_background(fig, 'constellations', dpi=300)

def plot_time_domain(x, y_norm, save_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x.real, label='Enviado real')
    ax.plot(y_norm.real, label='Recebido real')
    ax.set_title('Partes reais no tempo')
    ax.set_xlabel('Índice')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right')
    ax.grid(True)
    fig.tight_layout()
    save_fig_in_background(fig, 'real_time', dpi=300)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x.imag, label='Enviado imaginário')
    ax.plot(y_norm.imag, label='Recebido imaginário')
    ax.set_title('Partes imaginárias no tempo')
    ax.set_xlabel('Índice')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right')
    ax.grid(True)
    fig.tight_layout()
    save_fig_in_background(fig, 'imag_time', dpi=300)

def plot_magnitude_phase(x, y_norm, save_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.abs(x), label='Enviado magnitude')
    ax.plot(np.abs(y_norm), label='Recebido magnitude')
    ax.set_title('Magnitudes')
    ax.set_xlabel('Índice')
    ax.set_ylabel('Magnitude')
    ax.legend(loc='upper right')
    ax.grid(True)
    fig.tight_layout()
    save_fig_in_background(fig, 'magnitude', dpi=300)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.angle(x), label='Enviado fase')
    ax.plot(np.angle(y_norm), label='Recebido fase')
    ax.set_title('Fases')
    ax.set_xlabel('Índice')
    ax.set_ylabel('Fase (rad)')
    ax.legend(loc='upper right')
    ax.grid(True)
    fig.tight_layout()
    save_fig_in_background(fig, 'phase', dpi=300)

def plot_mid_samples(x_mid, y_mid, save_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_mid.real, label='Enviado real - meio')
    ax.plot(y_mid.real, label='Recebido real - meio')
    ax.set_title('Partes reais - amostras do meio')
    ax.set_xlabel('Índice')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right')
    ax.grid(True)
    fig.tight_layout()
    save_fig_in_background(fig, 'real_mid', dpi=300)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.abs(x_mid), label='Magnitude - meio')
    ax.plot(np.abs(y_mid), label='Magnitude recebida - meio')
    ax.set_title('Magnitudes - amostras do meio')
    ax.set_xlabel('Índice')
    ax.set_ylabel('Magnitude')
    ax.legend(loc='upper right')
    ax.grid(True)
    fig.tight_layout()
    save_fig_in_background(fig, 'magnitude_mid', dpi=300)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.angle(x_mid), label='Fase - meio')
    ax.plot(np.angle(y_mid), label='Fase recebida - meio')
    ax.set_title('Fases - amostras do meio')
    ax.set_xlabel('Índice')
    ax.set_ylabel('Fase (rad)')
    ax.legend(loc='upper right')
    ax.grid(True)
    fig.tight_layout()
    save_fig_in_background(fig, 'phase_mid', dpi=300)

def main():
    plot_constellations(IQ_x_complex, IQ_y_complex, IQ_y_complex_normalized, save_dir)
    plot_evm_windows_with_total(evm_windows, EVM_percentage_total, delta_evm_percentage, delta_evm_dB, window_size, 'Dataset', save_dir)
    plot_time_domain(IQ_x_complex, IQ_y_complex_normalized, save_dir)
    plot_magnitude_phase(IQ_x_complex, IQ_y_complex_normalized, save_dir)
    plot_mid_samples(IQ_x_complex_mid_samples, IQ_y_complex_normalized_mid_samples, save_dir)
    executor.shutdown(wait=True)
    print(f"Plots salvos no diretório '{save_dir}' no formato {export_format}.")

if __name__ == '__main__':
    main()

# Create a new directory for the files
directory = "IQ_data"
if not os.path.exists(directory):
    os.makedirs(directory)

# Function to save data in both .npy and .mat formats
def save_data(data, filename):
    np.save(os.path.join(directory, filename + '.npy'), data)
   # savemat(os.path.join(directory, filename + '.mat'), {filename: data})

# Separate I and Q components for the normalized signals
I_x = IQ_x_complex.real
Q_x = IQ_x_complex.imag
IQ_x_tuple = np.column_stack((I_x, Q_x))

I_y = IQ_y_complex.real
Q_y = IQ_y_complex.imag
IQ_y_tuple = np.column_stack((I_y, Q_y))

I_y2 = IQ_y_complex_normalized.real
Q_y2 = IQ_y_complex_normalized.imag
IQ_y_tuple2 = np.column_stack((I_y2, Q_y2))

# 2. Save as a tuple
save_data(IQ_x_tuple, 'sent_data_tuple')
save_data(IQ_y_tuple2, 'received_data_tuple_sync-phase')

# Estatísticas do array enviado
# Estatísticas do array enviado
# Estatísticas do array enviado
# Estatísticas do array enviado_path# Estatísticas do array enviado
# Estatísticas do array enviado# Estatísticas do array enviado
# Estatísticas do array enviado# Estatísticas do array enviado
# Estatísticas do array enviado# Estatísticas do array enviado
# Estatísticas do array enviado# Estatísticas do array enviado
# Estatísticas do array enviado# Estatísticas do array enviado
# Estatísticas do array enviado
sent_shape = IQ_x_tuple.shape
sent_dtype = IQ_x_tuple.dtype.name
I_x = IQ_x_tuple[:, 0]
Q_x = IQ_x_tuple[:, 1]
sent_I_stats = {
    'mean': float(I_x.mean()),
    'std':  float(I_x.std()),
    'min':  float(I_x.min()),
    'max':  float(I_x.max())
}
sent_Q_stats = {
    'mean': float(Q_x.mean()),
    'std':  float(Q_x.std()),
    'min':  float(Q_x.min()),
    'max':  float(Q_x.max())
}

# Estatísticas do array recebido
recv_shape = IQ_y_tuple2.shape
recv_dtype = IQ_y_tuple2.dtype.name
I_y = IQ_y_tuple2[:, 0]
Q_y = IQ_y_tuple2[:, 1]
recv_I_stats = {
    'mean': float(I_y.mean()),
    'std':  float(I_y.std()),
    'min':  float(I_y.min()),
    'max':  float(I_y.max())
}
recv_Q_stats = {
    'mean': float(Q_y.mean()),
    'std':  float(Q_y.std()),
    'min':  float(Q_y.min()),
    'max':  float(Q_y.max())
}

# Recalcule o fator de normalização (mesma fórmula de synchronize_and_normalize)
normalization_factor = float(np.sqrt(
    np.mean(np.abs(IQ_x_complex)**2) /
    np.mean(np.abs(IQ_y_complex_normalized)**2)
))

# Monte o dicionário de metadados
meta = {
    'sent': {
        'shape': list(sent_shape),
        'dtype': sent_dtype,
        'I_stats': sent_I_stats,
        'Q_stats': sent_Q_stats
    },
    'received': {
        'shape': list(recv_shape),
        'dtype': recv_dtype,
        'I_stats': recv_I_stats,
        'Q_stats': recv_Q_stats
    },
    'normalization_factor': normalization_factor,
    'EVM_percentage': float(evm_percentage),
    'EVM_dB': float(evm_dB),
    'noise_power': float(noise_power),
    'SNR_dB': float(SNR_dB),
    'samp_rate': float(parameters.get('samp_rate', 1.0)),
    'duration_s': float(parameters.get('duration_s', 0.0))
}

# Salva o JSON ao lado dos .npy em IQ_data/
meta_path = os.path.join(directory, 'metadata.json')
with open(meta_path, 'w', encoding='utf-8') as jf:
    json.dump(meta, jf, indent=2)
print(f"Metadados salvos em: {meta_path}")