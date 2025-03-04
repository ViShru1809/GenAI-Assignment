import numpy as np
import matplotlib.pyplot as plt

class dataGen:
    #Generating noisy linear data for regression analysis
    def __init__(self, x_values, true_slope, intercept=0.1, noise_level=1.0):
        self.x_values = x_values
        self.true_slope = true_slope
        self.intercept = intercept
        self.noise_level = noise_level
        self.y_values = self._noise_data_gen()

    def _noise_data_gen(self):
        #Generating y-values with noise
        noise = np.random.normal(0, self.noise_level, len(self.x_values))
        return self.true_slope * self.x_values + self.intercept + noise

    def get_data(self):
        #Return x_values and noisy y_values
        return self.x_values, self.y_values

    def plot_data(self):
        #Ploting generated noisy data and true line
        plt.scatter(self.x_values, self.y_values, label='Noisy Data', color='blue', alpha=0.5)
        plt.plot(self.x_values, self.true_slope * self.x_values + self.intercept,
                 label='True Line', color='red', linestyle='dashed')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title('Generated Dataset')
        plt.show()


class RegressionEstimator:       
    def __init__(self, x_values, y_values, intercept=0.1):
        self.x_values = x_values
        self.y_values = y_values
        self.intercept = intercept

    def brute_force(self, slope_range=(0, 5), steps=100):
        #Performing brute-force (ls) search to find the m1
        m1_values = np.linspace(*slope_range, steps)
        losses = [self._mse_loss(m) for m in m1_values]
        opti_m1 = m1_values[np.argmin(losses)]
        self._plot_loss_curve(m1_values, losses, opti_m1)

        return opti_m1

    def _mse_loss(self, m):
        #Calculating MSE loss for m
        predicted_y = m * self.x_values + self.intercept
        return np.mean((self.y_values - predicted_y) ** 2)

    def _plot_loss_curve(self, m1_values, losses, opti_m1):
        #Ploting loss curve for brute-force
        plt.plot(m1_values, losses, label="Loss Curve")
        plt.axvline(opti_m1, color='r', linestyle='dashed', label=f'Best m1: {opti_m1:.2f}')
        plt.xlabel('Slope (m1)')
        plt.ylabel('MSE Loss')
        plt.title('Brute-force Search for m1')
        plt.legend()
        plt.show()

    def grad_desc(self, learning_rate=0.01, max_iters=1000, tolerance=1e-6):
        #GD to estimate the m1
        estimated_m1 = np.random.uniform(0, 5)
        loss_history = []

        for i in range(max_iters):
            predicted_y = estimated_m1 * self.x_values + self.intercept
            gradient = -2 * np.mean(self.x_values * (self.y_values - predicted_y))
            estimated_m1 -= learning_rate * gradient
            loss = np.mean((self.y_values - predicted_y) ** 2)
            loss_history.append(loss)
            if len(loss_history) > 1 and abs(loss_history[-2] - loss_history[-1]) < tolerance:
                break

        self._loss_reduction(loss_history)
        return estimated_m1

    def _loss_reduction(self, loss_history):
        #Ploting loss reduction on iterations during gradient descent
        plt.plot(loss_history, label="Gradient Descent Loss")
        plt.xlabel("Iterations")
        plt.ylabel("MSE Loss")
        plt.title("Loss Reduction Using Gradient Descent")
        plt.legend()
        plt.show()


class regPpl:
    #combining data generation, regression estimation, and visualization
    def __init__(self, x_values, true_slope, intercept=0.1):
        self.data_generator = dataGen(x_values, true_slope, intercept)
        self.x_values, self.y_values = self.data_generator.get_data()
        self.regression_estimator = RegressionEstimator(self.x_values, self.y_values, intercept)

    def execute_brute_force(self):
        return self.regression_estimator.brute_force()
    def execute_grad_desc(self):
        return self.regression_estimator.grad_desc()
    def visualize_data(self):
        self.data_generator.plot_data()


#Main execution
if __name__ == "__main__":
    sample_size = 100
    x_values = np.linspace(-10, 10, sample_size)
    true_slope = 3.0
    intercept = 0.1    
    pipeline = regPpl(x_values, true_slope, intercept)
    pipeline.visualize_data()
    best_m1_brute = pipeline.execute_brute_force()
    best_m1_gd = pipeline.execute_grad_desc()

    #Comparing results
    print(f"Best m1 (Brute-force): {best_m1_brute:.5f}")
    print(f"Best m1 (Gradient Descent): {best_m1_gd:.5f}")
