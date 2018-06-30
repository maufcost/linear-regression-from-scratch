# Univariate Linear Regression Implementation from scratch using just numpy.
# Linear equation based on: y = m * x + b, which is the same as h = theta1 * x + theta0
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use("fivethirtyeight")

DATASET_PATH = "data.csv"

class LinearRegressionModel():
    """
    Univariate linear regression model classifier.
    """

    def __init__(self, dataset, learning_rate=0.001, num_iterations=100):
        """
        Class constructor.
        'dataset' is the numpy array equivalent of dataset points.
        'learning_rate' is alpha.
        """
        self.dataset = dataset
        self.b = 0  # Initial guess value for 'b'.
        self.m = 0  # Initial guess value for 'm'.
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.M = len(self.dataset)
        self.total_error = 0
        self.errors = [] # Error values for the graph's y-axis.

    def apply_gradient_descent(self):
        """
        Runs the gradient descent step 'num_iterations' times.
        """
        for i in range(self.num_iterations):
            self._do_gradient_step()
            self.errors.append(self._compute_error())

    def _do_gradient_step(self):
        """
        Performs each step of gradient descent, tweaking 'b' and 'm'.
        """
        b_summation = 0
        m_summation = 0
        # Doing the summation here.
        for i in range(self.M):
            x_value = self.dataset.iloc[i, 0]
            y_value = self.dataset.iloc[i, 1]
            b_summation += (((self.m * x_value) + self.b) - y_value) # * 1
            m_summation += (((self.m * x_value) + self.b) - y_value) * x_value

        # Updating parameter values 'b' and 'm'.
        self.b = self.b - (self.learning_rate * (1/self.M) * b_summation)
        self.m = self.m - (self.learning_rate * (1/self.M) * m_summation)
        # At this point. Gradient descent is finished.

    def _compute_error(self):
        """
        Computes the total error based on the linear regression cost function.
        """
        self.total_error = 0
        # Summation part of the cost function equation.
        for i in range(self.M):
            x_value = self.dataset.iloc[i, 0]
            y_value = self.dataset.iloc[i, 1]
            hypothesis = (self.m * x_value) + self.b
            self.total_error += ((hypothesis - y_value) ** 2)

        # The rest of the cost function equation.
        return (1 / (2 * self.M)) * self.total_error

    def __str__(self):
        return "Results: b: {}, m: {}, Final Total error: {}".format(round(self.b, 2), round(self.m, 2), round(self._compute_error(), 2))

    def get_prediction_based_on(self, x):
        """ Yields prediction, y, based on feature x. This function should only be
        used after this classifier has been trained. """
        return round(float((self.m * x) + self.b), 2) # Original type: Numpy float.

    def plot(self):
        """ Plots cost function error as a function of the number of iterations. """
        plt.plot([i for i in range(self.num_iterations)], self.errors, label="Error")
        plt.xlabel("Number of iterations")
        plt.ylabel("Cost Function Error")
        plt.title("Cost function error as a \nfunction of the number of iterations.")
        plt.legend()
        plt.show()

def main():

    # Loading dataset.
    school_dataset = pd.read_csv(DATASET_PATH)

    # Creating 'LinearRegressionModel' object.
    lr = LinearRegressionModel(school_dataset, 0.0001, 1000)

    # Applying gradient descent.
    lr.apply_gradient_descent()

    # Getting some predictions.
    hours = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for hour in hours:
        print("Studied {} hours and got {} points.".format(hour, lr.get_prediction_based_on(hour)))

    # Printing the class attribute values.
    print(lr)

    # Plotting cost function error as a function of the number of iterations.
    lr.plot()

if __name__ == "__main__":
    main()
