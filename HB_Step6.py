import numpy as np
from numpy import log10, abs, pi, sqrt
from scipy import signal  # Correct import for the 'signal' module
import sys
sys.path.append('./functions')  # Add the folder to the Python path if needed
from func import *

def main():

    ##########Needed parameters##############################################################
    # n = 4 #here n is the number of MZIs, (n-1) of them have 2\Delta*L arm length difference and 1 \Delta*L
    # N = 2 * n - 1
    # center_wvlengths = 1311 # choose of amounts from this array: [ 1271, 1291, 1311, 1333] (nm)
    # fs = FSR(center_wvlengths)
    # atten =50
    # del_p = 10 ** (1.0 - (atten / 20.0))#sqrt(2.54e-3) # page 147 Madsen book
    #######To get the result of the paper enable the amount of this part######################
    fs, N, del_p, center_wvlengths = 100e9, 7, sqrt(2.54e-3), 1550 #At 1550 nm, 100 GHz corresponds to \Delta\lambda = 0.8 nm.
    ###########################################################################################
    print(del_p)

    all_h = []
    labels = []
    ratios = []
    s = [4, 5, 6, 7, 8]
    # Loop through different values of N
    for s in s:
        m = 2 * s - 1
        G_z_3, G_z_2, GGr, HHr = half_band_trick(m, del_p, fs, plot=False)
        roots = poly_roots(GGr)
        rootG = pick_roots(roots, m, plot=False)
        G, H, G1, H1 = G_H_G1_H1(rootG)
        # G = lptobp(m, G, center_wvlengths)
        result_table(G, H, G1, H1, G_z_3, m)
        w, h = signal.freqz(G, [1], worN=2000, fs=fs)
        all_h.append(h)
        labels.append(f"N={N}")  # Add label for current N
        idx_1dB = np.where(20 * np.log10(np.abs(h)) < -1 )
        idx_20dB = np.where(20 * np.log10(np.abs(h)) < -20 )

        # Calculate the difference in frequency between these points
        width_1dB_to_20dB = np.abs(w[idx_1dB][0] / w[idx_20dB][0])
        print('N=', N, ',w[idx_1dB][0] = ', w[idx_1dB][0]/1e9, ',w[idx_20dB][0] = ', w[idx_20dB][0]/1e9, ',ratio = ', width_1dB_to_20dB )
        # Calculate ratio and append to ratios list
        ratios.append(width_1dB_to_20dB)

        # Plot all frequency responses on a single plot
    # Plot all frequency responses on a single plot
    plotly_response(center_wvlengths, w, all_h, labels, ratios)

if __name__ == '__main__':
    main()