README

Title:
Soniom HRTF Dataset - Snags list

Authors:
Jakub Sztandera

Description:
This archive contains a list of HRTF measurements identified as affected by measurement issues. The metrics used for identification purposes are outlined below. The analysis used two different versions of each individual HRTF. The free-field compensated version (_FreeFieldComp_48kHz) was used for calculation of the channel-wise RMS magnitude in dB. The raw recordings (_Raw_48kHz) were used to extract all other metrics. 

Metrics
ILD_Hem_Corr: Spearman correlation between horizontal-plane ILD values in the left and right hemifields.
Min_ILD_AzFront: Azimuth containing the lowest ILD value detected within 45:315°.
Min_ILD_AzBack: Azimuth containing the lowest ILD value detected within 225:135°.
ITD_Hem_Corr: Spearman correlation between horizontal-plane ITD values in the left and right hemifields.
Min_ITD_AzFront: Azimuth containing the lowest ITD value detected within 45:315°.
Min_ITD_AzBack: Azimuth containing the lowest ITD value detected within 225:135°.
Max_ITD_AzLeft: Azimuth containing the highest ITD value detected within 45:135°.
Max_ITD_AzRight: Azimuth containing the highest ITD value detected within 225:315°.
ITD_BadChangeRateWindows: Percentage of windows in which the ITD change range exceeded 200 µs, computed using a moving window size of 4.
RightLeft_Diff_Az0E0: RMS power difference between left and right channel calculated at Az = 0°, El = 0°. Expressed in dBFS.
ILD_Zscore_HemCorr: Z-score normalized ILD_Hem_Corr values computed using the dataset mean and standard deviation.
ITD_Zscore_HemCorr: Z-score normalized ITD_Hem_Corr values computed using the dataset mean and standard deviation.
AbsZscoreDiff_ITD_ILD: Cue-balance anomaly score computed from the distribution of the normalized ILD and ITD correlation values; larger deviations indicate more atypical measurements.
 
QC thresholds:
Values exceeding 3 for ILD_Zscore_HemCorr,ITD_Zscore_HemCorr, AbsZscoreDiff_ITD_ILD.
Values exceeding 5dB for RightLeft_Diff_Az0E0.


Contents:
- Outlier_[Date].csv contains the list of the outlying HRTF measurements. Outliers were signified as those whose target metrics exceeded 3SD from the mean distribution of the entire HRTF dataset (at the time of the analysis).  

Reproduction Instructions:
[Pending]

Run scripts in the following order:
[Pending]

Data Notes:
Missing values are coded as NA.

