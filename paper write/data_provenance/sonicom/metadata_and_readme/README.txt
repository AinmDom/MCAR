   _____  ____  _   _ _____ _____ ____  __  __ 
  / ____|/ __ \| \ | |_   _/ ____/ __ \|  \/  |
 | (___ | |  | |  \| | | || |   | |  | | \  / |
  \___ \| |  | | . ` | | || |   | |  | | |\/| |
  ____) | |__| | |\  |_| || |___| |__| | |  | |
 |_____/ \____/|_| \_|_____\_____\____/|_|  |_|
                                               
This folder contains individual Head Related Transfer Function (HRTF) measurements and 3D head scans from the SONICOM HRTF database.

Folder structure:

 - '3DSCAN' contains a 3D scan of the head and torso:

        - 'PXXXX.stl' is the result of the scan as a 3D mesh. You can import this file into nearly all 3D modelling software.
	  Some minor automated patching has been performed, but if you want to use it, for instance for 3D printing, 
          additional post-processing to fill in the holes is advised.
        - 'PXXXX_Project1.asc' contains a point cloud, an intermediate storage format consisting of individual 3D points before they are joined into a mesh.

 - 'HRTF' contains the audio measurements including raw audio, settings, log files and a helper MATLAB script to regenerate the HRTF files:

        - 'HRTF' subfolder contains processed HRTF files at several sample rates (44.1 kHz, 48 kHz, and 96 kHz, divided into individual folders).
          The files are stored in SOFA (https://www.sofaconventions.org) and 3dti formats. Descriptions are included below.

                * FOR BEGINNERS: we recommend trying 'PXXXX_FreeFieldCompMinPhase_NoITD_44kHz.3dti-hrtf' in the 3DTI Binaural Test Application 
                  (https://github.com/3DTune-In/3dti_AudioToolkit/releases/latest).

                - 'PXXXX_Raw_XXkHz': raw measured HRTF (50 ms long, no fade in/out applied);
                - 'PXXXX_Windowed_XXkHz': same as Raw, but windowed to 5 ms and with fade in/out applied;
                - 'PXXXX_FreeFieldComp_XXkHz': same as Windowed, but free-field compensation is applied via a linear-phase EQ filter;
                - 'PXXXX_FreeFieldCompMinPhase_XXkHz': same as Windowed, but free-field compensation is applied via a minimum-phase EQ filter.

                For each file, there is also a 'NoITD' version which has the ITDs removed and is compatible with the 3DTI Toolkit.
                This was done by time-shifting each measurement so that all the onsets were aligned. 
                The amount of shifted samples is stored as metadata in the SOFA file.

        - 'HPEQ' subfolder contains personal equalisation for Sennheiser HD 650 headphones in 44.1 kHz, 48 kHz, and 96 kHz sample rates.
                - 'PXXXX_headphoneEQ_XXkHz.mat': contains data of 5 headphone measurements and minimum-phase EQ filters for both ears;
                - 'PXXXX_headphoneEQ_XXkHz.wav': audio file with the impulse response of a single-channel minimum-phase EQ filter 
                                                 (average of left and right magnitude responses).

        - DO NOT USE 'run_this_to_generate_sofa.m' UNLESS YOU KNOW WHAT THE SCRIPT DOES. 
          It is a MATLAB script which was used to generate HRTF files from the raw measurement data in the folder. 
          Its functionality requires access to the Auditory Modelling Toolbox (https://www.amtoolbox.org) 
          and the reference measurement for the free-field compensation.
          All the HRTF files have already been pre-generated and located in the correct subfolders, so the script may overwrite the existing data!

- 'SYNTHETIC_HRTF' contains the data to create a synthetic HRTF as well as the synthesised HRTFs themselves 
	
	- ' PXXXX_preprocessed.stl' is the processed version of the watertight stl from the 3D scan folder. 
	The scans have been aligned to the Frankfurt plane, beheaded, smoothed and any hair removed. 
	Use this scan if you would like to perform your own adjustment to the ear canal.
	
	- 'PXXXX_plugged.stl' is the processed version but in addition the entrance to the ear canal is blocked/smoothed as the synthesised HRTFs are measured from the entrance of the ear canal.
	Use this scan if you would like to perform your own meshgrading.
	
	- 'PXXXX_graded_left/right.stl' is the graded meshes of the plugged mesh where the resolution of the mesh is high at the ipsilateral ear to improve computational efficiency of MESH2HRTF
	Use these scans if you would like to simulate the HRTFs using MESH2HRTF.

	- 'HRIR_SONICOM_44100/48000.sofa' is the output of MESH2HRTF - the synthesised HRTF - over the SONICOM evaluation grid. 
	Use these files if you would like to perform your own ITD removal/windowed or other postprocessing.

	- 'PXXXX_Measured/Synthetic_Windowed_NoITD_Scaled.sofa' is the postprocessed synthetic HRTF (and acoustically measured counterpart) with the ITD removed, windowed to 5ms and with the HRIRs level normalised such that the 0,0 position equates in RMS level to the measured and the KEMAR large ears HRTF at the 0,0 position. 
	Use these files if you would like to directly compare the synthesised and measured HRTFs numerically and perceptually. 

	Note: All scans may need to be slightly realigned if using the automatic ear index finder in mesh2hrtf or for comparison between models (e.g. parametric pinna modelling) as scans are currently approximately aligned to the entrance of the ear canal for grading and may need more accurate alignment for these applications based on the required reference point

- 'PHOTOGRAMMETRY' contains 72 photos in HEIC format taken all around the head at 5 degree intervals. 
  The other files contain a depth map (TIF file) and camera orientation (TXT file) for each picture.

------------------------------------------
Audio Experience Design, Dyson School of Design Engineering, Imperial College London
https://www.axdesign.co.uk

README v0.1, 16/03/2022
