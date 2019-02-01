% SNratio This computes the signal to noise ratio.
%
% SN = SNratio(f,p)
%
% Author Adrian Chan
%
% This computes the signal-to-noise ratio (SNR). Noise is defined as any
% unidentifiable high frequency component. The EMG spectrum is limited to
% 1000 Hz. The upper 20% of the frequency range is used to computed the
% noise (ensuring all frequencies are above 500 Hz). The noise is the 
% average of all power densities in the upper 20% frequency range is summed
% across the entire frequency range (i.e. take the average and muliply by
% the total number of frequency components). The SNR is ratio of the sum of
% all power densities and the noise.
%
% While the SNR should be sensitive to both low and high frequency noise,
% it is only sensitive to high frequency noise by definition. If there
% exists low frequency noise, this can result in falsely high SNR values.
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
%    SN: signal-to-noise ratio(can be converted to dB by using
%        10*log10(SNratio(f,P))
%
% Modifications
% 09/09/21 AC First created.
function SN = SNratio(f,p)

debugmode = false;

% remmove frequencies above 1000 Hz
index_below_1000 = find(f <= 1000);
f = f(index_below_1000);
p = p(index_below_1000);

N = round(length(f)/5); % 20% of the frequencies
index_f_upper = (length(f) - N):length(f);

% ensure that the upper frequencies are above 500 Hz
index_f_upper = index_f_upper(find(f(index_f_upper) > 500));
N = length(index_f_upper);

if debugmode == true
    figure
    plot(f,p,f(index_f_upper),p(index_f_upper),'ro');
    xlabel('f');
    ylabel('PSD');
    legend('PSD','Upper frequencies');
    title('SN ratio')
end

noise_power = sum(p(index_f_upper))/N*length(p);
total_power = sum(p);

SN = total_power/noise_power;