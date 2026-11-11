% preprocess_check.m - MATLAB-syntax cross-check of the Python PPG pipeline.
% Same 4th-order Butterworth band-pass (0.5-8 Hz, zero-phase filtfilt),
% z-score, and systolic peak detection (min distance 0.33 s, prominence 0.5)
% as src/preprocess.py. Runs in MATLAB (Signal Processing Toolbox) or GNU
% Octave (signal package). Reads results/matlab_windows.csv (20 raw PPG
% windows, 125 Hz) and writes results/matlab_filtered.csv and
% results/matlab_peaks.csv for scripts/compare_matlab.py.
%   Run from the repo root:  octave --no-gui matlab/preprocess_check.m
if exist('OCTAVE_VERSION', 'builtin')
  try
    pkg load signal
  catch
    % pixi conda-forge octave-signal is not registered with pkg: add its paths
    pfx = getenv('CONDA_PREFIX');
    addpath(genpath(fullfile(pfx, 'share', 'octave', 'packages')));
    addpath(genpath(fullfile(pfx, 'lib', 'octave', 'packages')));
  end
end
fs = 125;
X = csvread('results/matlab_windows.csv');
[b, a] = butter(4, [0.5 8] / (fs / 2));
F = zeros(size(X));
npk = zeros(size(X, 1), 1);
fid = fopen('results/matlab_peaks.csv', 'w');
for i = 1:size(X, 1)
  y = filtfilt(b, a, X(i, :));
  y = (y - mean(y)) / std(y, 1);          % population std, like numpy
  F(i, :) = y;
  locs = find_peaks_scipy(y, floor(0.33 * fs), 0.5);  % same as scipy distance=int(0.33*fs)
  npk(i) = numel(locs);
  fprintf(fid, '%d,', i - 1); fprintf(fid, '%d ', locs - 1); fprintf(fid, '\n');  % 0-based like Python
end
fclose(fid);
dlmwrite('results/matlab_filtered.csv', F, 'precision', '%.8f');
fprintf('windows=%d  mean peaks per window=%.2f\n', size(X, 1), mean(npk));
