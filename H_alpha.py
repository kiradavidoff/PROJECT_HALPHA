from astropy.io import fits
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit


def dark():
    """
    Loads dark frame FITS files from the '/data/Darks' directory.

    Returns:
        dark_files (list): List of FITS file objects for dark frames.
        dark_data (list): List of data arrays from dark frame FITS files.
        dark_names (list): List of filenames of the dark frames.
    """
    home_dir = os.getcwd()+'/data/Darks'
    files = os.listdir(home_dir)
    dark_names = [f for f in files if f.endswith(('.fits.gz'))]
    dark_files = [(fits.open(f'{home_dir}/{dark_names[i]}')[0]) for i in range(len(dark_names))]
    dark_data = [file.data for file in dark_files]

    return dark_files, dark_data, dark_names


def bias():
    """
    Loads bias frame FITS files from the '/data/Bias' directory.

    Returns:
        bias_files (list): List of FITS file objects for bias frames.
        bias_data (list): List of data arrays from bias frame FITS files.
        bias_names (list): List of filenames of the bias frames.
    """
    home_dir = os.getcwd()+'/data/Bias'
    files = os.listdir(home_dir)
    bias_names = [f for f in files if f.endswith(('.fits.gz'))]
    bias_files = [(fits.open(f'{home_dir}/{bias_names[i]}')[0]) for i in range(len(bias_names))]
    bias_data = [file.data for file in bias_files]

    return bias_files, bias_data, bias_names


def lights():
    """
    Loads light frame FITS files from the '/data/Lights' directory.

    Returns:
        lights_files (list): List of FITS file objects for light frames.
        lights_data (list): List of data arrays from light frame FITS files.
        lights_names (list): List of filenames of the light frames.
    """
    home_dir = os.getcwd()+'/data/Lights'
    files = sorted(os.listdir(home_dir))
    lights_names = [f for f in files if f.endswith(('.fits.gz'))]
    lights_files = [(fits.open(f'{home_dir}/{lights_names[i]}')[0]) for i in range(len(lights_names))]
    lights_data = [file.data for file in lights_files]

    return lights_files, lights_data, lights_names


def average_image(image_list):
    """
    Computes the average of a list of images.

    Args:
        image_list (list): List of 2D numpy arrays representing images.

    Returns:
        np.ndarray: The averaged image.
    """
    return np.mean(image_list, axis=0)


def corrected_image(l, d_avg, b_avg):
    """
    Corrects a light frame by subtracting dark and bias averages.

    Args:
        l (np.ndarray): Light frame data.
        d_avg (np.ndarray): Dark frame average data.
        b_avg (np.ndarray): Bias frame average data.

    Returns:
        np.ndarray: Corrected light frame data.
    """
    return l - d_avg - b_avg


def quickplot(input_image, name):
    """
    Displays a quick plot of an input image using a defined color scale.

    Args:
        input_image (np.ndarray): 2D array of the image data.
        name (str): Title of the plot.
    """
    displo = np.median(input_image) - np.std(input_image)
    disphi = np.median(input_image) + 2 * np.std(input_image)
    plt.title(name)
    plt.imshow(input_image, cmap='gray', vmin=displo, vmax=disphi)


def quickstats(input_image):
    """
    Prints basic statistics (min, max, mean, median, and standard deviation) of an image.

    Args:
        input_image (np.ndarray): 2D array of the image data.
    """
    print('Min:', np.min(input_image))
    print('Max:', np.max(input_image))
    print('Mean:', np.mean(input_image))
    print('Median:', np.median(input_image))
    print('Stdev:', np.std(input_image))


def header_info(fitsfile):
    """
    Prints header information from a FITS file, such as image type, CCD temperature, exposure time, and filter.

    Args:
        fitsfile (astropy.io.fits.HDUList): The FITS file to extract header info from.
    """
    image_head = fitsfile[0].header
    image_data = fitsfile[0].data

    print('Image type      = ', image_head['IMAGETYP'])
    print('CCD temperature = ', image_head['CCD-TEMP'], ' degrees C')
    print('Exposure time   = ', image_head['EXPTIME'], ' seconds')

    # check whether FILTER exists in the header first
    key = 'FILTER'
    if key in image_head.keys():
        print('Filter          = ', image_head['FILTER'])
    else:
        print('Filter          =  None')


def outliers_replace_median(y, clip=3, windowSize=50, plot=False):
    """
    Rolling median filter: for each point, checks if it's more than 'clip' standard deviations away from the local median
    using a window of 'windowSize' before and after. Instead of removing outliers, it replaces them with the local median.

    Args:
        y (array-like): y data.
        clip (float): Number of SDs to clip to.
        windowSize (int): How many points before and after to consider.
        plot (bool): If True, plot outliers.

    Returns:
        yModified (array-like): y with outliers replaced.
    """
    x = np.arange(len(y))  # Generate x values
    yModified = y.copy()  # Copy original data for modification

    for i in range(len(y)):
        # Define rolling window (excluding current point)
        start = max(0, i - windowSize)
        end = min(len(y), i + windowSize + 1)
        window = np.concatenate((y[start:i], y[i+1:end]))

        # Skip if not enough points
        if len(window) < 2:
            continue

        # Compute statistics
        median = np.median(window)
        std = np.std(window)

        # Detect and replace outliers
        if y[i] > (median + clip * std) or y[i] < (median - clip * std):
            yModified[i] = median  # Replace outlier with local median
            if plot:
                plt.plot(x[i], y[i], 'rx', label="Outlier" if i == 0 else "")

    # Plot results
    if plot:
        plt.plot(x, y, 'k-', alpha=0.4, label="Original Data")
        plt.plot(x, yModified, 'b-', label="Corrected Data")
        plt.legend()
        plt.grid()
        plt.show()

    return yModified





def gaussian(x, a, x0, sigma, c):
    """
    Calculates Gaussian y-values from input x-values given gaussian parameters.

    Args:
        x (array-like): x-values.
        a (float): Amplitude.
        x0 (float): Mean.
        sigma (float): Standard Deviation.
        c (float): Offset.

    Returns:
        y (array-like): Gaussian y-values.
    """
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + c



def outliers_replace_median_with_uncertainty(y, uncertainties, clip=3, windowSize=40, plot=False):
    """
    Rolling median filter with uncertainty propagation: for each point, checks if it's more than
    'clip' standard deviations away from the local median using a window of 'windowSize' before and after.
    Instead of removing outliers, it replaces them with the local median and propagates the uncertainty.

    Args:
        y (array-like): y data (values)
        uncertainties (array-like): corresponding uncertainties for each data point
        clip (float): Number of SDs to clip to
        windowSize (int): How many points before and after to consider
        plot (bool): If True, plot outliers

    Returns:
        yModified (array-like): y with outliers replaced
        uncertaintiesModified (array-like): uncertainty values after modification
    """

    x = np.arange(len(y))  # Generate x values
    yModified = y.copy()  # Copy original data for modification
    uncertaintiesModified = uncertainties.copy()  # Copy uncertainties for modification

    for i in range(len(y)):
        # Define rolling window (excluding current point)
        start = max(0, i - windowSize)
        end = min(len(y), i + windowSize + 1)
        window = np.concatenate((y[start:i], y[i+1:end]))
        window_uncertainties = np.concatenate((uncertainties[start:i], uncertainties[i+1:end]))

        # Skip if not enough points
        if len(window) < 2:
            continue

        # Compute statistics
        median = np.median(window)
        std = np.std(window)
        median_uncertainty = np.std(window_uncertainties) / np.sqrt(len(window_uncertainties))  # Uncertainty of the median estimate

        # Detect and replace outliers
        if y[i] > (median + clip * std) or y[i] < (median - clip * std):
            yModified[i] = median  # Replace outlier with local median
            uncertaintiesModified[i] = median_uncertainty  # Replace uncertainty with new uncertainty estimate
            if plot:
                plt.plot(x[i], y[i], 'rx', label="Outlier" if i == 0 else "")

    # Plot results
    if plot:
        plt.plot(x, y, 'k-', alpha=0.4, label="Original Data")
        plt.plot(x, yModified, 'b-', label="Corrected Data")
        plt.legend()
        plt.grid()
        plt.show()

    return yModified, uncertaintiesModified


def MedianClipRolling(y, yError, clip=3, windowSize=5):
    """
    Rolling sigma clip: for each point, checks if it's more than clip standard deviations
    away from the local mean, using a window of 'windowSize' before and after.

    Args:
        y (array-like): y data
        yError (array-like): error values for the y data
        clip (float): Number of SDs to clip to
        windowSize (int): How many points before and after to consider

    Returns:
        yClipped (array-like): y values with outliers set to local mean
        yErrorClipped (array-like): error values after clipping
    """

    yClipped = np.empty_like(y)
    yErrorClipped = np.empty_like(yError)

    for i in range(len(y)):
        # Rolling window excluding current point
        start = max(0, i - windowSize)
        end = min(len(y), i + windowSize + 1)

        # Create window excluding current point
        window = np.concatenate((y[start:i], y[i+1:end]))
        windowError = np.concatenate((yError[start:i], yError[i+1:end]))

        # When on the edge, skip
        if len(window) < 2:
            yClipped[i] = y[i]
            yErrorClipped[i] = yError[i]
            continue

        # Calculate local median and MAD
        windowMedian = np.median(window)
        MAD = np.mean(np.abs(window - windowMedian))

        mean = np.mean(window)
        std = np.std(window)

        # Remove Outliers
        if np.abs(y[i] - windowMedian) > clip * MAD:
                yClipped[i] = windowMedian
                yErrorClipped[i] = 1.253 * (np.sqrt(np.sum(windowError**2)) / len(window))
        else:
            yClipped[i] = y[i]
            yErrorClipped[i] = yError[i]

    return yClipped, yErrorClipped


def Normalise(y, err, xMin1, xMin2, plot=False):
    """
    Normalize the data by fitting a 3rd-degree polynomial to the continuum and then dividing the
    data by the fitted continuum. Error propagation is also computed for the normalization process.

    Args:
        y (array-like): Data to be normalized
        err (array-like): Error values associated with the data
        xMin1 (float): The lower bound of the fitting region
        xMin2 (float): The upper bound of the fitting region
        plot (bool): If True, will plot the original data, continuum, and the normalized data

    Returns:
        continuumNormalised (array-like): The data normalized by the continuum
        sigma_continuumNormalised (array-like): The uncertainty in the normalized data
    """
    x = np.linspace(0, len(y), len(y))
    ind = np.where((x < xMin1) | (x > xMin2))
    err_cor = err[ind]
    xValuesMaxs = x[ind]
    continuumMaxs = y[ind]

    coefsContinuum, cov = np.polyfit(xValuesMaxs, continuumMaxs, deg=3, w=1 / err_cor**2, cov=True)
    FitContinuum = np.poly1d(coefsContinuum)

    if plot:
        plt.plot(x, y, label='1D Spectrum')
        plt.plot(x[np.where(x < xMin1)], y[np.where(x < xMin1)], 'r', label='Selected area')
        plt.plot(x[np.where(x > xMin2)], y[np.where(x > xMin2)], 'r')
        smoothness = 1000
        xFitContinuum = np.linspace(np.min(x), np.max(x), smoothness)
        yFitContinuum = FitContinuum(xFitContinuum)

        plt.plot(xFitContinuum, yFitContinuum, label='3rd degree polynomial fit')
        plt.xlabel('X (pixels)')
        plt.ylabel('Flux (Sum of Y pixels)')
        plt.grid()
        plt.legend()
        plt.show()

    error_param = np.sqrt(np.diag(cov))

    err_a = x**3 * error_param[0]
    err_b = x**2 * error_param[1]
    err_c = x * error_param[2]
    err_d = error_param[3]

    # Propagate errors using the covariance matrix
    sigma_f = err_a**2 + err_b**2 + err_c**2 + err_d**2

    # Take square root of the total propagated error
    sigma_f = np.sqrt(sigma_f)
    yFitContinuum = FitContinuum(x)
    continuumNormalised = y / yFitContinuum
    sigma_continuumNormalised = continuumNormalised * np.sqrt((err / y)**2 + (sigma_f / yFitContinuum)**2)

    return continuumNormalised, sigma_continuumNormalised


def Lorentz(x, l0, l1, l2, l3):
    """
    Calculates the Lorentzian function for a given x value and its parameters.

    Args:
        x (float): The input x value
        l0 (float): Parameter that represents the offset
        l1 (float): Parameter that represents the amplitude
        l2 (float): Parameter that represents the center
        l3 (float): Parameter that represents the width

    Returns:
        v (float): The Lorentzian function value for the given parameters
    """
    L = l0 + (l1) / (1 + ((x - l2) / l3)**2)
    return L


def parameter_error(red_x2_array, parameter_array, parameter):
    """
    Returns the uncertainty in the selected parameter by changing the chi-squared value.

    Args:
        red_x2_array (array-like): Array of reduced chi-squared values for different parameter variations
        parameter_array (array-like): Array of parameter values that were varied
        parameter (float): The optimized parameter value

    Returns:
        delta (float): The uncertainty in the optimized parameter
    """
    mm = np.min(red_x2_array)
    thres2 = np.where(red_x2_array < (mm + 1))
    delta = np.abs(parameter_array[np.max(thres2)] - parameter)
    return delta


def chi_square_calc(image_data, error_image, fitted_line, red_x2_array, degree_of_freedom):
    """
    Computes the reduced chi-squared value for a fit.

    Args:
        image_data (array-like): The data to be fitted
        error_image (array-like): The uncertainty associated with the image data
        fitted_line (array-like): The model of the fitted line
        red_x2_array (list): An empty array to store the reduced chi-squared values
        degree_of_freedom (int): The degrees of freedom

    Returns:
        red_x2_array (list): The updated array containing reduced chi-squared values for each variation
    """
    x2 = np.sum(((image_data - fitted_line) / error_image)**2)
    red_x2 = x2 / degree_of_freedom
    red_x2_array.append(red_x2)
    return red_x2_array


def compute_param_uncertainty(index, delta_range, step, xx, data, error, fit_parameter):
    """
    Calculates the uncertainty in one of the parameters by varying it over a range.

    Args:
        index (int): Index of the parameter (0 = Ie, 1 = Re, 2 = n)
        delta_range (float): The range over which to vary the parameter
        step (float): The step size between parameter variations
        xx (array-like): The radius array of the Sersic profile
        data (array-like): The Sersic profile data
        error (array-like): The error in the profile
        fit_parameter (array-like): The optimized fit parameters

    Returns:
        uncertainty (float): The computed uncertainty of the parameter
    """
    # Calculate degrees of freedom (DOF)
    dof = len(xx) - len(fit_parameter)

    # Create an array of variations for the parameter
    base_value = fit_parameter[index]
    parameter_range = np.arange(base_value - delta_range, base_value + delta_range, step)

    # Empty array for chi-squared values
    chi_sq = []

    # Loop over the parameter range
    for val in parameter_range:
        # Change the variable parameter
        parameter = fit_parameter.copy()
        parameter[index] = val

        # Fit the Sersic model using the updated parameters
        y_ = Lorentz(xx, parameter[0], parameter[1], parameter[2], parameter[3])

        # Calculate the chi-squared value
        chi_sq = chi_square_calc(data, error, y_, chi_sq, dof)

    # Calculate the uncertainty from the chi-squared values
    uncertainty = parameter_error(chi_sq, parameter_range, base_value)
    return uncertainty


# Stats functions defined

def residuals(y, y_pred):
    """
    Compute the standard residuals between observed and predicted values.

    Parameters:
    y (array-like): Observed data points.
    y_pred (array-like): Predicted data points from the fitted model.

    Returns:
    numpy.ndarray: Residuals (difference between observed and predicted values).
    """
    return y - y_pred

def residuals_norm(residual,err):
    """
    Normalizes residuals by dividing them by their standard deviation.

    Parameters:
    residual (array-like): Residuals from a regression model.

    Returns:
    array-like: Normalized residuals.
    """
    return residual/err    # the error is approximated by doing the standard deviation of the residuals

def chi2(residuals,err):
    """
    Compute the chi-square statistic from residuals.

    Parameters:
    residuals (array-like): The residuals (differences between observed and predicted values).

    Returns:
    float: The chi-square statistic, assuming uniform uncertainty estimated by standard deviation.
    """
    return sum((residuals / err) ** 2)

def chi2_red(residuals, n_params,err):
    """
    Compute the reduced chi-square statistic.

    Parameters:
    residuals (array-like): The residuals (differences between observed and predicted values).
    sample_norm (array-like): The sample dataset (used to determine the number of data points).
    n_params (int): The number of free parameters in the model.

    Returns:
    float: The reduced chi-square statistic, normalized by degrees of freedom.
    """
    return chi2(residuals,err) / (residuals.shape[0] - n_params)


def WavelengthConverter(x, feb = False, march = False,err=False):
    """
    args:
        x (array-like): x input
        feb (bool): If the iamge was taken 25.02.2025
        march (bool): If the iamge was taken 05.03.2025

    returns:
        wavelengths (array-like): Wavelength values for each x

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

    # Calculate Wavelengths
    wavelengths = ((a * (xFlipped**2)) + (b * xFlipped) + c)/m
    if err== True:
        errs= xFlipped*((2*a * (xFlipped)) + b )/m
        return errs
    return wavelengths

def lorentzian(x, l0, l1, l2, l3):
    """a function that creates a lorentzian function from x and set of parameters

    input: x(function variable), l0 = offset level, l1 = amplitude of line,
           l2 = line position, l3 = derivative to give half-width

    output: the shape of lorentzian function for given parameters"""
    L = l0 +  ( l1 / ( 1 + ((x - l2)/l3)**2 ))
    return L

def lorentzian_jacobian(x, l0, l1, l2, l3):
    # Ensure x is a NumPy array
    x = np.atleast_1d(x)
    A = 1 + ((x - l2) / l3)**2

    # Make sure each derivative has the same shape as x
    dL_dl0 = np.ones_like(x)
    dL_dl1 = 1.0 / A
    dL_dl2 = (2 * l1 * (x - l2)) / (l3**2 * A**2)
    dL_dl3 = (2 * l1 * (x - l2)**2) / (l3**3 * A**2)

    # Stack them into an array of shape (len(x), 4)
    return np.vstack((dL_dl0, dL_dl1, dL_dl2, dL_dl3)).T


def compute_error_in_spectrum(xx,parameters_list, parameter_uncertainty, parameter_cov):
    Range = np.arange(0,2048,1)
    error_y = []
    for c in Range:
        # Assume x_array[c] is your array of x values for column c (e.g. shape (69,))
        # and parameter[c] is a 4-element array [l0, l1, l2, l3] for that column.
        x_vals = xx[c]  # shape (69,)
        params = parameters_list[c]  # shape (4,)

        # Build the Jacobian for all x values; modify your lorentzian_jacobian to return an array of shape (N,4)
        J = lorentzian_jacobian(x_vals, params[0], params[1], params[2], params[3])  # shape (69,4)

        # If you only have diagonal uncertainties, create a diagonal covariance matrix:
        cov_diag = np.diag(parameter_uncertainty[c])  # shape (4,4)

        # Compute the propagated uncertainty at each x:
        #sigma_y_array = np.array([np.sqrt(J_row.T @ cov_matrix[c] @ J_row) for J_row in J])
        sigma_y_array = np.array([np.sqrt(J.T @ parameter_cov[c] @ J) for J in J])

        error_y.append(sigma_y_array)

    #_range = np.arange(0,69,1)
    ymax_l0 = []
    ymax_error = []

    for f in Range:
        error_y_l0 =np.sqrt(np.abs(error_y[f]**2 + parameters_list[f][0]**2))
        ymax_l0.append(error_y_l0)

    error_ymax1_sum = []
    for g in Range:
        #error_y_max1_sq = ymax_l0[g]**2
        #error_ymax1_sum.append(error_y_max1_sq)
        error_ymax_each = np.sqrt(np.abs(np.sum(ymax_l0[g]**2)))
        ymax_error.append(error_ymax_each)

    return ymax_error

def Lorenzian_1D_spectrum(image_subset,bg11,bg12,bg21,bg22, image_error):
    Range = np.arange(0,2048,1)

    y_max1 = []
    y_integral = []
    y_1 = []
    x_array = []
    parameter = []
    image_data_array = []
    sigma_column_array = []
    cov_matrix = []
    parameter_uncertainty_array = []

    for a in Range:
        # slice the image
        image_slice1 = image_subset[bg11:bg22,a:a+1]

        # background
        background11 = image_slice1[bg11:bg12,0:1]
        background22 = image_slice1[bg21:bg22, 0:1]
        background_combined1 = np.concatenate((background11.flatten(), background22.flatten()))
        bakcground_mean1 = np.mean(background_combined1)
        error_bg = np.std(background_combined1)
        #error_bg = bakcground_mean1/ (len(background_combined1))

        image_slice_bgs1 = image_slice1 - bakcground_mean1

        error_image_slice_pixel = np.sqrt(np.abs((image_slice_bgs1.flatten()  + image_error[a]**2 + error_bg**2)))
        #error_image_slice_pixel = np.sqrt(np.abs((image_slice_bgs1.flatten()  + image_error[a]**2)))
        sigma_column_array.append(error_image_slice_pixel)

        error_im = np.sqrt(np.abs(image_slice1))

        image_data_array.append(image_slice_bgs1.flatten())

        print(sigma_column_array)
        xx = np.arange(0,len(image_slice_bgs1),1)
        x_array.append(xx)

        capped_max = np.minimum(np.max(image_slice_bgs1), 5000) # to avoid hot pixels
        #capped_min = np.maximum(np.min(image_slice_bgs1), -30)

        #po = np.asarray([np.min(image_slice_bgs1), capped_max, np.median(xx), 8], dtype=float)
        po = np.asarray([0,capped_max, np.median(xx), 8], dtype=float)
        #po = np.asarray([capped_min, capped_max, np.median(xx), 8], dtype=float)

        line_par1 , line_cov1 = optimize.curve_fit(lorentzian, xx, image_slice_bgs1.flatten(), po, sigma = error_image_slice_pixel,maxfev = 1000000)
        parameter.append(line_par1)
        param_err = np.sqrt(np.diag(line_cov1))
        cov_matrix.append(line_cov1)
        parameter_uncertainty_array.append(param_err)

        y1 = lorentzian(xx,line_par1[0],line_par1[1],line_par1[2],line_par1[3])
        # the offset is given by l0 - lline_par1[0]
        sum1 = np.sum(y1)
        max1 = np.max(y1)
        #sum1 = np.sum(y1[bg12:bg21])
        #sum1 = np.sum(y1 - line_par1[0])
        y_max1.append(max1)
        y_integral.append(sum1)

    return y_max1,y_integral, parameter, parameter_uncertainty_array, cov_matrix, image_data_array


def sigma_dbl():
    """Finds error on the Dark, Bias and Light frame"""
    sigma_dark = np.sqrt(np.mean(Dark_images_bs, axis=0)) / np.sqrt(np.abs(len(Dark_images_bs)))
    sigma_Bias = np.std(Bias_images,axis=0) / np.sqrt(len(Bias_images))
    sigma_signal= np.sqrt(np.abs(Light_images_correct[image_number]))
    sigma_dark_cor= np.sqrt(sigma_dark**2+sigma_Bias**2)
    noise= np.sqrt(sigma_dark_cor**2+sigma_Bias**2+sigma_signal**2)
    return noise

def back(noise):
    """Find the noise in background and propogates noise"""


    sigma_b1= noise[513:596,0:2048][10:24]
    sigma_b2= noise[513:596,0:2048][63:77]

    sigmab = np.concatenate([sigma_b1, sigma_b2], axis=0)





    propagated_error =np.sqrt( np.mean(sigmab)**2 + noise[513:596,0:2048]**2)
    noise_sqrt=np.sqrt(np.sum(propagated_error**2,axis=0))/len(np.sqrt(np.sum(propagated_error**2,axis=0)))
    return noise_sqrt/np.sqrt(2048)


def img():
    """removes the background from subset"""
    background_1= np.mean(Light_images_correct[image_number][513:596,0:2048][10:24])
    background_2= np.mean(Light_images_correct[image_number][513:596,0:2048][63:77])

    alf_Dra_p002_02= Light_images_correct[image_number][520:590,0:2048] - (background_1+ background_2)/2
    return np.sum(alf_Dra_p002_02, axis=0)



def sigma_dbl():
    """Calculates The initial Noise on the Image"""
    sigma_dark = np.sqrt(np.mean(Dark_images_bs, axis=0)) / np.sqrt(np.abs(len(Dark_images_bs)))
    sigma_Bias = np.std(Bias_images,axis=0) / np.sqrt(len(Bias_images))
    sigma_signal= np.sqrt(np.abs(Light_images_correct[image_number]))
    sigma_dark_cor= np.sqrt(sigma_dark**2+sigma_Bias**2)
    noise= np.sqrt(sigma_dark_cor**2+sigma_Bias**2+sigma_signal**2)
    return noise

def back(noise):
    """Removes the Background of the image"""


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
            popt, pcov = optimize.curve_fit(Lorentz, x_range, sample_nw, p0=initial_guess, maxfev=10000, sigma=sigma_range)
            line_shape = Lorentz(x, *popt)  # Fix: use `x_range` instead of undefined `x_slice`

            # Extract fitted parameters
            l0, l1, l2, l3 = popt

            # Store results
            H_alpha.append(l1)
            H_err.append(np.sqrt(np.diag(pcov))[1])
            chi2_left.append(chi2_red(residuals(continium_normarlised, line_shape), 4, normalised_err))

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
            popt, pcov = optimize.curve_fit(Lorentz, x_range, sample_nw, p0=initial_guess, maxfev=10000, sigma=sigma_range)
            line_shaped = Lorentz(x, *popt)  # Fix: use `x_range` instead of undefined `x_slice`

            # Extract fitted parameters
            l0, l1, l2, l3 = popt

            # Store results
            H_alpha.append(l1)
            H_err.append(np.sqrt(np.diag(pcov))[1])
            chi2_right.append(chi2_red(residuals(continium_normarlised, line_shaped), 4, normalised_err))

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
            popt, pcov = optimize.curve_fit(Lorentz, x_range, sample_nw, p0=initial_guess, maxfev=10000, sigma=sigma_range)
            line_shaped = Lorentz(x, *popt)  # Fix: use `x_range` instead of undefined `x_slice`

            # Extract fitted parameters
            l0, l1, l2, l3 = popt

            # Store results
            H_alpha.append(l1)
            H_err.append(np.sqrt(np.diag(pcov))[1])
            chi2_increase.append(chi2_red(residuals(continium_normarlised, line_shaped), 4, normalised_err))

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
            popt, pcov = optimize.curve_fit(Lorentz, x_range, sample_nw, p0=initial_guess, maxfev=10000, sigma=sigma_range)
            line_shaped = Lorentz(x, *popt)  # Fix: use `x_range` instead of undefined `x_slice`

            # Extract fitted parameters
            l0, l1, l2, l3 = popt

            # Store results
            H_alpha.append(l1)
            H_err.append(np.sqrt(np.diag(pcov))[1])
            chi2_decrease.append(chi2_red(residuals(continium_normarlised, line_shaped), 4, normalised_err))

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



    popt, pcov = optimize.curve_fit(Lorentz, x_range, sample_nw, p0=initial_guess, maxfev=10000, sigma=sigma_range)


    return popt, pcov, sample_nw, sigma_range, x_range

def final(popt, pcov, sample_nw, sigma_range, x_range):
    amplitude_cov = np.sqrt(np.diag(pcov))[1]
    position_cov = np.sqrt(np.diag(pcov))[2]

    amplitude, position= popt[1],popt[2]
    amplitude_err,position_err= compute_param_uncertainty(1, 50,0.0001, x_range, sample_nw, sigma_range, popt),compute_param_uncertainty(2, 50,0.001, x_range, sample_nw, sigma_range, popt)

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
    """Calculates Chi for double lorentzian model"""

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
            chi2_left.append(chi2_red(residuals(continium_normarlised, line_shape), 4, normalised_err))

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
            chi2_right.append(chi2_red(residuals(continium_normarlised, line_shaped), 4, normalised_err))

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
            chi2_increase.append(chi2_red(residuals(continium_normarlised, line_shaped), 4, normalised_err))

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
            chi2_decrease.append(chi2_red(residuals(continium_normarlised, line_shaped), 4, normalised_err))

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
        chi_sq = chi_square_calc(data, error, y_,chi_sq, 7)


    # Use parameter error to compute the uncertainty
    uncertainty = parameter_error(chi_sq, parameter_range, base_value)
    return uncertainty



def sn_dl(continium_normarlised,normalised_err):
    """Finds the signal to noise ratio"""
    x,y=np.arange(0, len(continium_normarlised)), continium_normarlised


    # Fit the Double Lorentzian
    popt, pcov = curve_fit(double_lorentz, x, y, p0=[-0.7, 1100, 10, -0.7, 1100, 10, 10], sigma=normalised_err, maxfev=10000)



    line_shaped= double_lorentz(x,*popt)
    amplitude=popt[0]+popt[3]

    err=np.sqrt(compute_param_uncertainty_double_lorentz(0, 50, 0.001, x, y, normalised_err, popt)**2+compute_param_uncertainty_double_lorentz(3, 50, 0.001, x, y, normalised_err, popt)**2)
    print(chi2_red(residuals(continium_normarlised, line_shaped), 7, normalised_err))

    return amplitude, err, popt, pcov

def pos_dl(continium_normarlised, normalised_err):
    """Finds the position of H alpha in the conitnium"""
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
def final(popt, pcov, sample_nw, sigma_range, x_range):
    """Prints required values for single lorentzian"""

    amplitude_cov = np.sqrt(np.diag(pcov))[1]
    position_cov = np.sqrt(np.diag(pcov))[2]

    amplitude, position= popt[1],popt[2]
    amplitude_err,position_err= compute_param_uncertainty(1, 50,0.0001, x_range, sample_nw, sigma_range, popt),ha.compute_param_uncertainty(2, 50,0.001, x_range, sample_nw, sigma_range, popt)

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
    """Double lorentzian which is the sum of tzo lorentzian + a offset"""
    lorentz1 = amp1 / (1 + ((x - center1) / width1) ** 2)
    lorentz2 = amp2 / (1 + ((x - center2) / width2) ** 2)
    return lorentz1 + lorentz2 + offset

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
        chi_sq = chi_square_calc(data, error, y_,chi_sq, 7)


    # Use parameter error to compute the uncertainty
    uncertainty = parameter_error(chi_sq, parameter_range, base_value)
    return uncertainty


def sn(continium_normarlised,normalised_err):
    """Prints S/N, amplitude with error"""
    x,y=np.arange(0, len(continium_normarlised)), continium_normarlised


    # Fit the Double Lorentzian
    popt, _ = optimize.curve_fit(double_lorentz, x, y, p0=[-0.7, 1100, 10, -0.7, 1100, 10, 10], sigma=normalised_err, maxfev=10000)



    line_shaped= double_lorentz(x,*popt)
    amplitude=popt[0]+popt[3]

    err=np.sqrt(compute_param_uncertainty_double_lorentz(0, 50, 0.001, x, y, normalised_err, popt)**2+compute_param_uncertainty_double_lorentz(3, 50, 0.001, x, y, normalised_err, popt)**2)
    print(chi2_red(residuals(continium_normarlised, line_shaped), 7, normalised_err))
    print(f'amplitude = {amplitude}+-{err}')
    print(f'SNR= {amplitude/err}')
    return amplitude/err

def pos(continium_normarlised, normalised_err):
    """Prints position with error"""
    x,y=np.arange(0, len(continium_normarlised)), continium_normarlised


    # Fit the Double Lorentzian
    popt, _ = optimize.curve_fit(double_lorentz, x, y, p0=[-0.7, 1100, 10, -0.7, 1100, 10, 10], sigma=normalised_err,maxfev=10000)
    err=np.sqrt((compute_param_uncertainty_double_lorentz(1, 50, 0.001, x, y, normalised_err, popt)**2+compute_param_uncertainty_double_lorentz(4, 50, 0.001, x, y, normalised_err, popt)**2)/4)
    print(f'position= {(popt[1]+popt[4])/2}+-{err}')
    return (popt[1]+popt[4])/2



