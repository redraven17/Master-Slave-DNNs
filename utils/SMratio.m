% SMratio This computes the signal-to-motion artifact ratio.
%
% SM = SMratio(f,p)
%
% Author Adrian Chan
%
% This computes the signal-to-motion artifact ratio (SMR). The SMR is based
% on two main assumptions: 1) frequency of motion artifacts are below 20
% Hz, and 2) the shapes of the non-contaminated EMG power spectrum is
% fairly linear below 20 Hz. The SMR is the ratio of the sum of all power
% densities for all frequencies below 600 Hz and the sum of all power
% densities below 20 Hz that exceed a straight line between the axis origin
% and the highest mean power density above 35 Hz. The mean power density is
% computed by averaged 13 consecutive points in the EMG power spectrum.
%
% Reference: Sinderby C, Lindstrom L, Grassino AE, "Automatic assessment of
% electromyogram quality", Journal of Applied Physiology, vol. 79, no. 5,
% pp. 1803-1815, 1995.
%
% Inputs
%    f: frequencies (Hz)
%    p: power spectral density values
%
% Outputs
%    SM: signal-to-motion artifact (can be converted to dB by using
%        10*log10(SMratio(f,P))
%
% Modifications
% 09/09/21 AC First created.
function SM = SMratio(f,p)

debugmode = false;

% remove frequencies above 600 Hz
index_below_600 = find(f <= 600);
f = f(index_below_600);
p = p(index_below_600);

% average PSD over N points using N/2 points before and after
N = 13;
b = ones(N,1)/N;
a = 1;
mean_psd = filter(b,a,[p;zeros(floor(N/2),1)]);
mean_psd = mean_psd(floor(N/2) + (1:length(f)));

if debugmode == true
    figure
    plot(f,p,f,mean_psd);
    xlabel('f');
    ylabel('PSD');
    legend('PSD','Averaged PSD');
    title('SM ratio')
end

index_f_above_35 = find(f > 35);
highest_mean_psd = max(mean_psd(index_f_above_35));
index_highest_mean_psd = find(mean_psd == highest_mean_psd);
f_highest_mean_psd = f(index_highest_mean_psd);

if debugmode == true
    hold on, plot(f_highest_mean_psd,highest_mean_psd,'ro'), hold off
end

index_f_below_20 = find(f <= 20);
slope = highest_mean_psd/f_highest_mean_psd;
index_exceed_line = find(p(index_f_below_20) > (slope*f(index_f_below_20)));
power_above_line = sum(p(index_exceed_line));

if debugmode == true
    hold on, plot([0,f_highest_mean_psd],[0,highest_mean_psd],'r:'), hold off
    hold on, plot(f(index_exceed_line),p(index_exceed_line),'rx'), hold off
end

M0 = sum(p);

SM = M0/power_above_line;
