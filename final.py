import numpy as np
import matplotlib.pyplot as plt
import statistics as stats
from scipy import optimize
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.rcParams['figure.figsize'] = 10,10
import astropy
from astropy.io import fits
import os
import  H_alpha as ha




def sigma_dbl():
    """Calculate noise of the image"""
    sigma_dark = np.sqrt(np.mean(Dark_images_bs, axis=0)) / np.sqrt(np.abs(len(Dark_images_bs)))
    sigma_Bias = np.std(Bias_images,axis=0) / np.sqrt(len(Bias_images))
    sigma_signal= np.sqrt(np.abs(Light_images_correct[image_number]))
    sigma_dark_cor= np.sqrt(sigma_dark**2+sigma_Bias**2)
    noise= np.sqrt(sigma_dark_cor**2+sigma_Bias**2+sigma_signal**2)
    return noise

def back(noise):
    """Removes background noise of the image"""


    sigma_b1= noise[513:596,0:2048][10:24]
    sigma_b2= noise[513:596,0:2048][63:77]

    sigmab = np.concatenate([sigma_b1, sigma_b2], axis=0)





    propagated_error =np.sqrt( np.mean(sigmab)**2 + noise[513:596,0:2048]**2)
    noise_sqrt=np.sqrt(np.sum(propagated_error**2,axis=0))/len(np.sqrt(np.sum(propagated_error**2,axis=0)))
    return noise_sqrt/np.sqrt(2048)


def img():
    background_1= np.mean(Light_images_correct[image_number][513:596,0:2048][10:24])
    background_2= np.mean(Light_images_correct[image_number][513:596,0:2048][63:77])

    alf_Dra_p002_02= Light_images_correct[image_number][520:590,0:2048] - (background_1+ background_2)/2
    return np.sum(alf_Dra_p002_02, axis=0)


def chi(continium_normarlised,normalised_err):
    # Initialize lists to store results
    H_alpha = []
    H_err = []
    chi2_left = []
    chi2_right = []
    chi2_increase = []
    chi2_decrease = []



    # Ensure x values are properly initialized
    x = np.arange(0, len(continium_normarlised))

    # First loop: shifting range leftward
    for i in range(500):
        # Select data range
        mask = (x > 600 - i) & (x < 1600 - i)
        sample_nw = continium_normarlised[mask]
        x_range = x[mask]
        sigma_range = normalised_err[mask]

        # Initial guess for Lorentzian parameters: [Amplitude, Peak Position, Width, Offset]
        initial_guess = [-1, -3, 1000, 5]

        # Fit Lorentzian curve
        try:
            popt, pcov = optimize.curve_fit(ha.Lorentz, x_range, sample_nw, p0=initial_guess, maxfev=10000, sigma=sigma_range)
            line_shape = ha.Lorentz(x, *popt)  # Fix: use `x_range` instead of undefined `x_slice`

            # Extract fitted parameters
            l0, l1, l2, l3 = popt

            # Store results
            H_alpha.append(l1)
            H_err.append(np.sqrt(np.diag(pcov))[1])
            chi2_left.append(ha.chi2_red(ha.residuals(continium_normarlised, line_shape), 4, normalised_err))

        except RuntimeError:
            print(f"Fit failed at iteration {i}")

    # Second loop: shifting range rightward
    for i in range(500):
        # Select data range
        mask = (x > (600 + i)) & (x < (1600 + i))
        sample_nw = continium_normarlised[mask]
        x_range = x[mask]
        sigma_range = normalised_err[mask]

        # Initial guess for Lorentzian parameters


        # Fit Lorentzian curve
        try:
            popt, pcov = optimize.curve_fit(ha.Lorentz, x_range, sample_nw, p0=initial_guess, maxfev=10000, sigma=sigma_range)
            line_shaped = ha.Lorentz(x, *popt)  # Fix: use `x_range` instead of undefined `x_slice`

            # Extract fitted parameters
            l0, l1, l2, l3 = popt

            # Store results
            H_alpha.append(l1)
            H_err.append(np.sqrt(np.diag(pcov))[1])
            chi2_right.append(ha.chi2_red(ha.residuals(continium_normarlised, line_shaped), 4, normalised_err))

        except RuntimeError:
            print(f"Fit failed at iteration {i}")

    # First loop: increasing the range
    for i in range(len(continium_normarlised)):
        # Select data range
        mask = (x > 600 - i) & (x < 1600 + i)
        sample_nw = continium_normarlised[mask]
        x_range = x[mask]
        sigma_range = normalised_err[mask]

        # Initial guess for Lorentzian parameters: [Amplitude, Peak Position, Width, Offset]


        # Fit Lorentzian curve
        try:
            popt, pcov = optimize.curve_fit(ha.Lorentz, x_range, sample_nw, p0=initial_guess, maxfev=10000, sigma=sigma_range)
            line_shaped = ha.Lorentz(x, *popt)  # Fix: use `x_range` instead of undefined `x_slice`

            # Extract fitted parameters
            l0, l1, l2, l3 = popt

            # Store results
            H_alpha.append(l1)
            H_err.append(np.sqrt(np.diag(pcov))[1])
            chi2_increase.append(ha.chi2_red(ha.residuals(continium_normarlised, line_shaped), 4, normalised_err))

        except RuntimeError:
            print(f"Fit failed at iteration {i}")

    # Second loop: decreasing the range
    for i in range(400):
        # Select data range
        mask = (x > (600 + i)) & (x < (1600 - i))
        sample_nw = continium_normarlised[mask]
        x_range = x[mask]
        sigma_range = normalised_err[mask]

        # Initial guess for Lorentzian parameters


        # Fit Lorentzian curve
        try:
            popt, pcov = optimize.curve_fit(ha.Lorentz, x_range, sample_nw, p0=initial_guess, maxfev=10000, sigma=sigma_range)
            line_shaped = ha.Lorentz(x, *popt)  # Fix: use `x_range` instead of undefined `x_slice`

            # Extract fitted parameters
            l0, l1, l2, l3 = popt

            # Store results
            H_alpha.append(l1)
            H_err.append(np.sqrt(np.diag(pcov))[1])
            chi2_decrease.append(ha.chi2_red(ha.residuals(continium_normarlised, line_shaped), 4, normalised_err))

        except RuntimeError:
            print(f"Fit failed at iteration {i}")

    chi2_min=[min(chi2_increase),min(chi2_decrease), min(chi2_left),min(chi2_right)]
    if min(chi2_min)==chi2_min[0]:
        print('increase')
        mask = (x > 600 - np.where(np.array(chi2_increase)==np.min(chi2_increase))[0][0]) & ( x < 1600 + np.where(np.array(chi2_increase)==np.min(chi2_increase))[0][0])
        sample_nw = continium_normarlised[mask]
        x_range = x[mask]
        sigma_range = normalised_err[mask]

    if min(chi2_min)==chi2_min[1]:
        print('decrease')
        mask = (x > 600 - np.where(np.array(chi2_decrease)==np.min(chi2_decrease))[0][0]) & ( x < 1600 + np.where(np.array(chi2_decrease)==np.min(chi2_decrease))[0][0])
        sample_nw = continium_normarlised[mask]
        x_range = x[mask]
        sigma_range = normalised_err[mask]

    if min(chi2_min)==chi2_min[2]:
        print('left')
        mask = (x > 600 - np.where(np.array(chi2_left)==np.min(chi2_left))[0][0]) & ( x < 1600 + np.where(np.array(chi2_left)==np.min(chi2_left))[0][0])
        sample_nw = continium_normarlised[mask]
        x_range = x[mask]
        sigma_range = normalised_err[mask]

    if min(chi2_min)==chi2_min[-1]:
        print('right')
        mask = (x > 600 - np.where(np.array(chi2_right)==np.min(chi2_right))[0][0]) & ( x < 1600 + np.where(np.array(chi2_right)==np.min(chi2_right))[0][0])
        sample_nw = continium_normarlised[mask]
        x_range = x[mask]
        sigma_range = normalised_err[mask]

    # Initial guess for Lorentzian parameters: [Amplitude, Peak Position, Width, Offset]



    popt, pcov = optimize.curve_fit(ha.Lorentz, x_range, sample_nw, p0=initial_guess, maxfev=10000, sigma=sigma_range)


    return popt, pcov, sample_nw, sigma_range, x_range

def final(popt, pcov, sample_nw, sigma_range, x_range):
    amplitude_cov = np.sqrt(np.diag(pcov))[1]
    position_cov = np.sqrt(np.diag(pcov))[2]

    amplitude, position= popt[1],popt[2]
    amplitude_err,position_err= ha.compute_param_uncertainty(1, 50,0.0001, x_range, sample_nw, sigma_range, popt),ha.compute_param_uncertainty(2, 50,0.001, x_range, sample_nw, sigma_range, popt)

    positionLambda, positionLambdaError = WavelengthConverter(position, position_err, march = True)
    print('CHI')
    print(positionLambda, positionLambdaError)
    positionLambda, positionLambdaErrorCov = WavelengthConverter(position, position_cov, march = True )
    print('COV')
    print(positionLambda, positionLambdaErrorCov)


    print(f'ImageNumber = {1}\n\n')

    print('               VALUE              CHI SQUARED ERROR     COVARIANCE MATRIX ERROR')
    print(f"amplitude   {amplitude} +- {amplitude_err} +- {amplitude_cov}")
    print(f"position     {position} +- {position_err}    +- {position_cov}")
    print(f"\nposition (Angstrom):  {positionLambda}+- {positionLambdaError}+- {positionLambdaErrorCov}")


    print('\n')
    print(f'SN (CHI): {np.abs(amplitude/amplitude_err)}')
    print(f'SN (COV): {np.abs(amplitude/amplitude_cov)}')


def double_lorentz(x, amp1, center1, width1, amp2, center2, width2, offset):
    lorentz1 = amp1 / (1 + ((x - center1) / width1) ** 2)
    lorentz2 = amp2 / (1 + ((x - center2) / width2) ** 2)
    return lorentz1 + lorentz2 + offset

def chi_2(continium_normarlised,normalised_err):


    # Initialize lists to store results
    H_alpha = []
    H_err = []
    chi2_left = []
    chi2_right = []
    chi2_increase = []
    chi2_decrease = []

    initial_guess = [-0.5, 950, 100, -0.3, 1050, 120, 1]
    # Ensure x values are properly initialized
    x = np.arange(0, len(continium_normarlised))

    # First loop: shifting range leftward
    for i in range(500):
        # Select data range
        mask = (x > 600 - i) & (x < 1600 - i)
        sample_nw = continium_normarlised[mask]
        x_range = x[mask]
        sigma_range = normalised_err[mask]

        # Initial guess for Lorentzian parameters: [Amplitude, Peak Position, Width, Offset]
        initial_guess = [-0.5, 950, 100, -0.3, 1050, 120, 1]

        # Fit Lorentzian curve
        try:
            popt, pcov = optimize.curve_fit(double_lorentz, x_range, sample_nw, p0=initial_guess, maxfev=10000, sigma=sigma_range)
            line_shape = double_lorentz(x, *popt)  # Fix: use `x_range` instead of undefined `x_slice`



            # Store results

            H_err.append(np.sqrt(np.diag(pcov))[1])
            chi2_left.append(ha.chi2_red(ha.residuals(continium_normarlised, line_shape), 4, normalised_err))

        except RuntimeError:
            print(f"Fit failed at iteration {i}")

    # Second loop: shifting range rightward
    for i in range(500):
        # Select data range
        mask = (x > (600 + i)) & (x < (1600 + i))
        sample_nw = continium_normarlised[mask]
        x_range = x[mask]
        sigma_range = normalised_err[mask]

        # Initial guess for Lorentzian parameters


        # Fit Lorentzian curve
        try:
            popt, pcov = optimize.curve_fit(double_lorentz, x_range, sample_nw, p0=initial_guess, maxfev=10000, sigma=sigma_range)
            line_shaped = double_lorentz(x, *popt)  # Fix: use `x_range` instead of undefined `x_slice`



            H_err.append(np.sqrt(np.diag(pcov))[1])
            chi2_right.append(ha.chi2_red(ha.residuals(continium_normarlised, line_shaped), 4, normalised_err))

        except RuntimeError:
            print(f"Fit failed at iteration {i}")

    # First loop: increasing the range
    for i in range(len(continium_normarlised)):
        # Select data range
        mask = (x > 600 - i) & (x < 1600 + i)
        sample_nw = continium_normarlised[mask]
        x_range = x[mask]
        sigma_range = normalised_err[mask]

        # Initial guess for Lorentzian parameters: [Amplitude, Peak Position, Width, Offset]


        # Fit Lorentzian curve
        try:
            popt, pcov = optimize.curve_fit(double_lorentz, x_range, sample_nw, p0=initial_guess, maxfev=10000, sigma=sigma_range)
            line_shaped = double_lorentz(x, *popt)  # Fix: use `x_range` instead of undefined `x_slice`



            H_err.append(np.sqrt(np.diag(pcov))[1])
            chi2_increase.append(ha.chi2_red(ha.residuals(continium_normarlised, line_shaped), 4, normalised_err))

        except RuntimeError:
            print(f"Fit failed at iteration {i}")

    # Second loop: decreasing the range
    for i in range(400):
        # Select data range
        mask = (x > (600 + i)) & (x < (1600 - i))
        sample_nw = continium_normarlised[mask]
        x_range = x[mask]
        sigma_range = normalised_err[mask]

        # Initial guess for Lorentzian parameters


        # Fit Lorentzian curve
        try:
            popt, pcov = optimize.curve_fit(double_lorentz, x_range, sample_nw, p0=initial_guess, maxfev=10000, sigma=sigma_range)
            line_shaped = double_lorentz(x, *popt)  # Fix: use `x_range` instead of undefined `x_slice`




            H_err.append(np.sqrt(np.diag(pcov))[1])
            chi2_decrease.append(ha.chi2_red(ha.residuals(continium_normarlised, line_shaped), 4, normalised_err))

        except RuntimeError:
            print(f"Fit failed at iteration {i}")
    chi2_min=[min(chi2_increase),min(chi2_decrease), min(chi2_left),min(chi2_right)]
    if min(chi2_min)==chi2_min[0]:
        print('increase')
        mask = (x > 600 - np.where(np.array(chi2_increase)==np.min(chi2_increase))[0][0]) & ( x < 1600 + np.where(np.array(chi2_increase)==np.min(chi2_increase))[0][0])
        sample_nw = continium_normarlised[mask]
        x_range = x[mask]
        sigma_range = normalised_err[mask]

    if min(chi2_min)==chi2_min[1]:
        print('decrease')
        mask = (x > 600 - np.where(np.array(chi2_decrease)==np.min(chi2_decrease))[0][0]) & ( x < 1600 + np.where(np.array(chi2_decrease)==np.min(chi2_decrease))[0][0])
        sample_nw = continium_normarlised[mask]
        x_range = x[mask]
        sigma_range = normalised_err[mask]

    if min(chi2_min)==chi2_min[2]:
        print('left')
        mask = (x > 600 - np.where(np.array(chi2_left)==np.min(chi2_left))[0][0]) & ( x < 1600 + np.where(np.array(chi2_left)==np.min(chi2_left))[0][0])
        sample_nw = continium_normarlised[mask]
        x_range = x[mask]
        sigma_range = normalised_err[mask]

    if min(chi2_min)==chi2_min[-1]:
        print('right')
        mask = (x > 600 - np.where(np.array(chi2_right)==np.min(chi2_right))[0][0]) & ( x < 1600 + np.where(np.array(chi2_right)==np.min(chi2_right))[0][0])
        sample_nw = continium_normarlised[mask]
        x_range = x[mask]
        sigma_range = normalised_err[mask]



    popt, pcov = optimize.curve_fit(double_lorentz, x_range, sample_nw, p0=initial_guess, maxfev=10000, sigma=sigma_range)


    return popt, pcov, sample_nw, sigma_range, x_range

def compute_param_uncertainty_double_lorentz(index, delta_range, step, xx, data, error, fit_parameter):
    """
    Work out the uncertainty in one of the parameters by varying it over a range using a double Lorentzian fit.

    input:
    index: index of the parameter (0 = amp1, 1 = center1, 2 = width1, 3 = amp2, 4 = center2, 5 = width2, 6 = offset)
    delta_range: the range which parameter varies
    step: difference between each variation
    xx: the independent variable array (e.g., x-axis)
    data: observed data
    error: the error in the data
    fit_parameter: optimized parameters (amp1, center1, width1, amp2, center2, width2, offset)

    Returns:
    uncertainty: computed uncertainty of the parameter.
    """
    # Calculate degrees of freedom
    dof = len(xx) - len(fit_parameter)

    # Create an array of variation for the specified parameter
    base_value = fit_parameter[index]
    parameter_range = np.arange(base_value - delta_range, base_value + delta_range, step)

    # Empty list to store chi-square values
    chi_sq = []

    # Loop over the parameter range
    for val in parameter_range:
        # Change the variable parameter
        parameter = fit_parameter.copy()
        parameter[index] = val

        # Fit the model using the current parameters
        y_ = double_lorentz(xx, parameter[0], parameter[1], parameter[2], parameter[3], parameter[4], parameter[5], parameter[6])

        # Calculate chi-square
        chi_sq = ha.chi_square_calc(data, error, y_,chi_sq, 7)


    # Use parameter error to compute the uncertainty
    uncertainty = ha.parameter_error(chi_sq, parameter_range, base_value)
    return uncertainty



def sn_dl(continium_normarlised,normalised_err):
    x,y=np.arange(0, len(continium_normarlised)), continium_normarlised


    # Fit the Double Lorentzian
    popt, pcov = curve_fit(double_lorentz, x, y, p0=[-0.7, 1100, 10, -0.7, 1100, 10, 10], sigma=normalised_err, maxfev=10000)



    line_shaped= double_lorentz(x,*popt)
    amplitude=popt[0]+popt[3]

    err=np.sqrt(compute_param_uncertainty_double_lorentz(0, 50, 0.001, x, y, normalised_err, popt)**2+compute_param_uncertainty_double_lorentz(3, 50, 0.001, x, y, normalised_err, popt)**2)
    print(ha.chi2_red(ha.residuals(continium_normarlised, line_shaped), 7, normalised_err))

    return amplitude, err, popt, pcov

def pos_dl(continium_normarlised, normalised_err):
    x,y=np.arange(0, len(continium_normarlised)), continium_normarlised


    # Fit the Double Lorentzian
    popt,pcov = curve_fit(double_lorentz, x, y, p0=[-0.7, 1100, 10, -0.7, 1100, 10, 10], sigma=normalised_err,maxfev=10000)

    err=np.sqrt((compute_param_uncertainty_double_lorentz(1, 50, 0.001, x, y, normalised_err, popt)**2+compute_param_uncertainty_double_lorentz(4, 50, 0.001, x, y, normalised_err, popt)**2)/4)


    return (popt[1]+popt[4])/2, err, popt, pcov



def final_double_lorentz(amplitude, amplitude_err_chi, position, position_err_chi, pcov, image_number=1):
    """
    Mirror of your single-Lorentz 'final', but for two lines:
    - amplitude = A1 + A2
    - position  = (x01 + x02)/2
    Uses pcov to compute covariance-based errors on these combined quantities.
    """
    #  covariance-based uncertainties for combined params
    amp_cov_var = pcov[0,0] + pcov[3,3] + 2.0*pcov[0,3]
    amplitude_err_cov = np.sqrt(max(amp_cov_var, 0.0))

    pos_cov_var = (pcov[1,1] + pcov[4,4] + 2.0*pcov[1,4]) / 4.0
    position_err_cov = np.sqrt(max(pos_cov_var, 0.0))

    #  wavelength conversion (both error types)
    lam_chi, lam_err_chi = WavelengthConverter(position, position_err_chi, march=True)
    lam_cov, lam_err_cov = WavelengthConverter(position, position_err_cov, march=True)


    print(f'ImageNumber = {image_number}\n')
    print('               VALUE              CHI/MC ERROR           COVARIANCE-MATRIX ERROR')
    print(f"amplitude   {amplitude} +- {amplitude_err_chi}        +- {amplitude_err_cov}")
    print(f"position    {position} +- {position_err_chi}         +- {position_err_cov}")
    print(f"\nposition (Angstrom):  {lam_chi} +- {lam_err_chi}      +- {lam_err_cov}\n")

    # S/N both ways (use absolute to be safe with negative absorption depths)
    sn_chi = np.abs(amplitude/amplitude_err_chi) if amplitude_err_chi != 0 else np.inf
    sn_cov = np.abs(amplitude/amplitude_err_cov) if amplitude_err_cov != 0 else np.inf
    print(f'SN (CHI): {sn_chi}')
    print(f'SN (COV): {sn_cov}')


def WavelengthConverter(x,xError, feb = False, march = False):
    """
    Convert pixel positions to wavelengths using a quadratic dispersion relation.

    args:
        x (array-like): Pixel positions in the x direction.
        xError (array-like): Error on each x
        feb (bool): If the iamge was taken 25.02.2025
        march (bool): If the iamge was taken 05.03.2025

    returns:
        wavelengths (array-like): Wavelength values for each x
        wavelengthErrors (array-like): Wavelength errors for each x

    """
    # Params
    a = -1.8 * (10**(-4))
    c = 359230

    # Order number
    m = 54

    # Depending on date
    if feb and march:
        raise ValueError("Both 'feb' and 'march' cannot be True at the same time.")
    elif feb:
        b = -3.3079
    elif march:
        b = -3.3158
    else:
        raise ValueError("Either 'feb' or 'march' must be True.")

    # Horizontal Flip
    xFlipped = np.flip(x)
    xErrorFlipped = np.flip(xError)

    # Calculate Wavelengths
    wavelengths = ((a * (xFlipped**2)) + (b * xFlipped) + c)/m

    # Calculate Error:
    wavelengthErrors = (1/m)*((2*a*xFlipped) + (b))* xErrorFlipped


    return wavelengths, wavelengthErrors

if __name__=="__main__":

    Dark_files, Dark_images, Dark_names = ha.dark()
    Bias_files, Bias_images, Bias_names= ha.bias()
    Light_file, Light_images, Light_names= ha.lights()

        # subtract the mean bias from all frames
    Dark_images_bs = Dark_images - np.mean(Bias_images, axis=0)

    # substract the Dark frame from the Light images with bias subtracted
    Light_images_correct = Light_images - np.mean(Dark_images_bs, axis=0)- np.mean(Bias_images, axis=0)

    image_number=float(input("Image number (0 to 8): "))
    initial_guess = [-1, -3, 1000, 5]

    if image_number<6:
        noise_full_image=sigma_dbl()

        noise_subset =back(noise_full_image)

        image_subset =img()

        alf_Dra_p001_02_1D_corrected, noise_corrected=ha.MedianClipRolling(image_subset,noise_subset,clip=3, windowSize=25)

        continium_normarlisedd,normalised_errr= ha.Normalise( alf_Dra_p001_02_1D_corrected,noise_corrected, 700, 1300)

        popt_1, pcov_1, sample_nw_1, sigma_range_1, x_range_1 =chi(continium_normarlisedd,normalised_errr)

        final(popt_1, pcov_1, sample_nw_1, sigma_range_1, x_range_1)

    else:
        image_number=8

        noise_full_image=sigma_dbl()


        noise_subset =back(noise_full_image)

        image_subset =img()

        alf_Dra_p001_02_1D_corrected, noise_corrected=ha.MedianClipRolling(image_subset,noise_subset,clip=3, windowSize=25)


        continium_normarlisedd,normalised_errr= ha.Normalise( alf_Dra_p001_02_1D_corrected,noise_corrected, 700, 1300)

        amplitude, amplitude_err,popt_am,pcov_am=sn_dl(continium_normarlisedd,normalised_errr)

        position,pos_err,popt_pos,pcov_pos= pos_dl(continium_normarlisedd,normalised_errr)

        final_double_lorentz(amplitude, amplitude_err, position, pos_err, pcov_pos, image_number=8)
