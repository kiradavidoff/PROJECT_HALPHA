# Echelle visible magnitude limits on stellar Halpha S/N

## Description

The H-alpha order image was processed to correct for instrumental noise (Bias and Dark) and removed background noise. Then, was transformed into a one-dimensional (1D) array and normalised, producing a spectrum where the absorption feature appears as a distinct dip. The resulting spectra were then fitted around the dip to determine the amplitude and position of the absorption line, with estimated uncertainties. The extracted position was converted to wavelength. Finally, the amplitude and its uncertainty were used to define the signal and noise, allowing us to compute the S/N for each image and analyse its correlation with the star’s apparent magnitude.

## Repository structure

project-directory/
 │── Data/
     │── Bias/
         │── bias.fit.gz
         │──....
     │── Darks/
         │── dark.fit.gz
         │──....
     │── Lights/
         │── light.fit.gz
         │──....

 │── Notebooks/
      │── double_lorentzian.ipynb
      │── lorentzian.ipynb
      │── signal_to_noise.ipynb

 │── H_alpha.py
 │── results.py
 │── requirements.txt
 │── table.csv



## How to run code

`python H_ALPHA/results.py`



