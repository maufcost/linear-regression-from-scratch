# Univariate Linear Regression Implementation from scratch using just numpy.
# Linear equation based on: y = m * x + b, which is the same as h = theta1 * x + theta0
import numpy as np

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

    def apply_gradient_descent(self):
        """
        Runs the gradient descent step 'num_iterations' times.
        """
        for i in range(self.num_iterations):
            self._do_gradient_step()

    def _do_gradient_step(self):
        """
        Performs each step of gradient descent, tweaking 'b' and 'm'.
        """
        b_summation = 0
        m_summation = 0
        # Doing the summation here.
        for i in range(self.M):
            x_value = self.dataset[i, 0]
            y_value = self.dataset[i, 1]
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
        for i in range(self.M):
            x_value = self.dataset[i, 0]
            y_value = self.dataset[i, 1]
            self.total_error += ((self.m * x_value) + self.b) - y_value
        return self.total_error

    def __str__(self):
        return "Results: b: {}, m: {}, Final Total error: {}".format(round(self.b, 2), round(self.m, 2), round(self._compute_error(), 2))

    def get_prediction_based_on(self, x):
        return round(float((self.m * x) + self.b), 2) # Original type: Numpy float.

def main():

    # Loading dataset.
    school_dataset = np.genfromtxt(DATASET_PATH, delimiter=",")

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

if __name__ == "__main__":
    main()

# Later:
# Print a graph with the cost function (the error amount) as a function of 'm'.
# Print a graph with the cost function (the error amount) as a function of 'b'.
# Create another file for multivariate linear regression.
