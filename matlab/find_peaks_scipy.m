function locs = find_peaks_scipy(x, distance, min_prom)
% FIND_PEAKS_SCIPY  Peak finder matching scipy.signal.find_peaks(x,
% distance=distance, prominence=min_prom): strict local maxima, then the
% distance rule (taller peaks win), then the prominence rule (whole-signal
% bases). Written by hand because Octave's findpeaks has no prominence option.
  x = x(:)';
  n = numel(x);
  p = find(x(2:n-1) > x(1:n-2) & x(2:n-1) > x(3:n)) + 1;
  % distance: keep peaks in order of height, drop neighbours closer than distance
  keep = true(size(p));
  [~, order] = sort(x(p), 'descend');
  for k = order
    if ~keep(k), continue; end
    j = k - 1;
    while j >= 1 && p(k) - p(j) < distance, keep(j) = false; j = j - 1; end
    j = k + 1;
    while j <= numel(p) && p(j) - p(k) < distance, keep(j) = false; j = j + 1; end
  end
  p = p(keep);
  % prominence
  prom = zeros(size(p));
  for k = 1:numel(p)
    i = p(k);
    l = i; lmin = x(i);
    while l > 1 && x(l - 1) <= x(i), l = l - 1; lmin = min(lmin, x(l)); end
    r = i; rmin = x(i);
    while r < n && x(r + 1) <= x(i), r = r + 1; rmin = min(rmin, x(r)); end
    prom(k) = x(i) - max(lmin, rmin);
  end
  locs = p(prom >= min_prom);
end
