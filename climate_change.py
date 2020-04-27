import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import re

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAINING_INTERVAL = range(1961, 2008)
TESTING_INTERVAL = range(2008, 2017)

"""
Begin helper code
"""
class Dataset(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Dataset instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature

        f.close()

    def get_yearly_temp(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d numpy array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return np.array(temperatures)

    def get_daily_temp(self, city, month, day, year):
        """
        Get the daily temperature for the given city and time (year + date).

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified time (year +
            date) and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year {} is not available".format(year)
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]

def se_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d numpy array of values estimated by a linear
            regression model
        model: a numpy array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = np.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]

##########################
#    End helper code     #
##########################

def lin_regression(x, y):
    """
    Generate a linear regression model for the set of data points.

    Args:
        x: a list of length N, representing the x-coordinates of
            the N sample points
        y: a list of length N, representing the y-coordinates of
            the N sample points

    Returns:
        (m, b): A tuple containing the slope and y-intercept of the regression line,
                which are both floats.
    """
    # Calculate mean of x and y and initalize numerator and denominator of m 
    num_m = 0
    den_m = 0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    # Use formula to calculate m
    for i in range(len(x)):
        num_m += ((x[i]-x_mean)*(y[i]-y_mean))
        den_m += (x[i]-x_mean)**2
    m = num_m/den_m
    b = y_mean-(m*x_mean)
    return m, b

def get_total_squared_error(x, y, m, b):
    '''
    Calculate the squared error of the regression model given the set of data points.

    Args:
        x: a list of length N, representing the x-coordinates of
            the N sample points
        y: a list of length N, representing the y-coordinates of
            the N sample points
        m: The slope of the regression line
        b: The y-intercept of the regression line


    Returns:
        the total squared error of our regression
    '''
    total_error = 0
    for i in range(len(x)):
        total_error += (y[i] - (m * x[i] + b))**2
    return total_error

def make_models(x, y, degs):
    """
    Generate a polynomial regression model for each degree in degs, given the set
    of data points.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        degs: a list of integers that correspond to the degree of each polynomial
            model that will be fit to the data

    Returns:
        a list of numpy arrays, where each array is a 1-d numpy array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    coefficients = []
    for deg in degs:
        coefficients.append(np.polyfit(x, y, deg))
    return coefficients

def evaluate_models_on_training(x, y, models):
    """
    For each regression model, compute the R-squared value, and plot the data
    along with the best fit curve. For linear regression models (i.e. models with
    degree 1), you should also compute the SE/slope.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (i.e. the model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        R-squared of your model evaluated on the given data points,
        and SE/slope (if degree of this model is 1 -- see se_over_slope).

    R-squared and SE/slope should be rounded to 3 decimal places.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    # Sets font size to 9 to make title fit.
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 9}
    plt.rc('font', **font)
    # Array of sample y values.
    sample_y = np.array(y)
    # Iterates through model
    for model in models:
        plt.figure()
        predicted_y = []
        # Adds predicted y to list for each x value.
        for x_value in x:
            predicted_y.append(np.polyval(model, x_value))
        # Array of predicted y values
        y_array = np.array(predicted_y)
        # Calculate R-squared values
        R_squared = round(r2_score(sample_y, y_array), 3)
        # Plot labelled graph with blue points.
        plt.plot(x, sample_y, "bo")
        plt.xlabel("Year")
        plt.ylabel("Temperature ('C)")
        # Checks if degree 1 polynomial (i.e. 2 coefficients)
        if len(model) == 2:
            # Calculate Ratio to 3 decimal places
            SE_Slope_Ratio = round(se_over_slope(x,sample_y,y_array, model), 3)
            plt.title("Yearly Temperature for R-squared=" + str(R_squared) + ", 1st Degree, Standard Error : Slope=" + str(SE_Slope_Ratio))
        else:
            plt.title("Yearly Temperature for R-squared=" + str(R_squared) + ", Degree " + str(len(model)-1))
        plt.plot(x, y_array, "r")


def generate_cities_averages(temp, multi_cities, years):
    """
    For each year in the given range of years, computes the average of the
    annual temperatures in the given cities.

    Args:
        temp: instance of Dataset
        multi_cities: (list of str) the names of cities to include in the average
            annual temperature calculation
        years: (list of int) the range of years of the annual averaged temperatures

    Returns:
        a numpy 1-d array of floats with length = len(years). Each element in
        this array corresponds to the average annual temperature over the given
        cities for a given year.
    """
    average_annual_temperatures = []
    for year in years:
        annual_temperatures = []
        for city in multi_cities:
            annual_temperatures.append(temp.get_yearly_temp(city, year))
        average_annual_temperatures.append(np.mean(annual_temperatures))
    return np.array(average_annual_temperatures)


def find_interval(x, y, length, use_positive_slope):
    """
    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        length: the length of the interval
        use_positive_slope: a boolean whose value specifies whether to look for
            an interval with the most extreme positive slope (True) or the most
            extreme negative slope (False)

    Returns:
        a tuple of the form (i, j) such that the application of linear (deg=1)
        regression to the data in x[i:j], y[i:j] produces the most extreme
        slope and j-i = length.

        In the case of a tie, it returns the most recent interval. For example,
        if the intervals (2,5) and (8,11) both have the same slope, (8,11) should
        be returned.

        If such an interval does not exist, returns None
    """
    x = list(x)
    y = list(y)
    # Make list of x subsets of size = length
    x_sublists = []
    y_sublists = []
    ix = 0
    for x_value in x:
        x_sublists.append(x[ix:ix+length])
        if ix+length == len(x):
            break
        ix += 1
    iy = 0
    for y_value in x:
        y_sublists.append(y[iy:iy+length])
        if iy+length == len(y):
            break
        iy += 1
    # Now you have a list of "length" long lists of consecutive x values
    # and a list of "length" long lists of consecutive x values with corresponding
    # lists having the same index
    slopes_list = []
    max_slope_val = 0
    maxreti = 0
    minreti = 0
    min_slope_val = 0
    for i in range(len(x_sublists)):
        slope = lin_regression(x_sublists[i], y_sublists[i])[0]
        slopes_list.append(slope)
        if slope >= max_slope_val: 
            max_slope_val = slope
            maxreti = i
        if slope <= min_slope_val: 
            min_slope_val = slope
            minreti = i
        i += 1
    # If positive slope desired
    if use_positive_slope:
        return (maxreti, maxreti + length) if max_slope_val > 0 else None
    # If negative slope desired
    else: 
        return (minreti, minreti + length) if min_slope_val < 0 else None

def rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d numpy array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    N = len(y)
    sum_diff_sq = 0
    for i in range (N):
        diff = y[i] - estimated[i]
        diff_sq = diff**2
        sum_diff_sq += diff_sq
    return (sum_diff_sq/N)**.5


def evaluate_models_on_testing(x, y, models):
    """
    For each regression model, compute the RMSE for this model and plot the
    test data along with the model's estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points.

    RMSE should be rounded to 3 decimal places.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N test data sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N test data sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    # Sets font size to 9 to make title fit.
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 9}
    plt.rc('font', **font)
    # Array of sample y values.
    sample_y = np.array(y)
    # Iterates through model
    for model in models:
        plt.figure()
        predicted_y = []
        # Adds predicted y to list for each x value.
        for x_value in x:
            predicted_y.append(np.polyval(model, x_value))
        # Array of predicted y values
        y_array = np.array(predicted_y)
        # Calculate R-squared values
        rmnse = round(rmse(sample_y, y_array),3)
        # Plot labelled graph with blue points.
        plt.plot(x, sample_y, "bo")
        plt.xlabel("Year")
        plt.ylabel("Temperature ('C)")
        plt.title("Yearly Temperature for RMSE=" + str(rmnse) + ", Degree " + str(len(model)-1))
        plt.plot(x, y_array, "r")
        


if __name__ == '__main__':

    pass

    x = list(range(1961, 2017))
    y = []
    data_set = Dataset("data.csv")
    for year in x:
        y.append(data_set.get_daily_temp("PORTLAND", 12, 25, year))
    evaluate_models_on_training(np.array(x), np.array(y), make_models(np.array(x), np.array(y), [1]))
    x = list(range(1961, 2017))
    y = generate_cities_averages(data_set, ['PORTLAND'], x)
    evaluate_models_on_training(np.array(x), np.array(y), make_models(np.array(x), np.array(y), [1]))
    y2 = generate_cities_averages(data_set, ['SAN FRANCISCO'], x)
    start, end = find_interval(x, y2, length = 30, use_positive_slope = True)
    new_x = list(range(x[start], x[end]))
    new_y = generate_cities_averages(data_set, ['SAN FRANCISCO'], new_x)
    evaluate_models_on_training(np.array(new_x), np.array(new_y), make_models(np.array(new_x), np.array(new_y), [1]))
    y2 = generate_cities_averages(data_set, ['SAN FRANCISCO'], x)
    start, end = find_interval(x, y2, length = 20, use_positive_slope = False)
    new_x = list(range(x[start], x[end]))
    new_y = generate_cities_averages(data_set, ['SAN FRANCISCO'], new_x)
    evaluate_models_on_training(np.array(new_x), np.array(new_y), make_models(np.array(new_x), np.array(new_y), [1]))
    y3 = generate_cities_averages(data_set, CITIES, TRAINING_INTERVAL)
    x3 = list(TRAINING_INTERVAL)
    evaluate_models_on_training(np.array(x3), np.array(y3), make_models(np.array(x3), np.array(y3), [2,10]))
    y4 = generate_cities_averages(data_set, CITIES, TESTING_INTERVAL)
    x4 = list(TESTING_INTERVAL)
    evaluate_models_on_testing(np.array(x4), np.array(y4), make_models(np.array(x3), np.array(y3), [2,10]))
    