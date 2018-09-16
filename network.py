import numpy as np



class NeuralNetwork:
    def __init__(self):
        # Seeding helps to get consistent results everytime this network is run.
        np.random.seed(1)

        # this creates an array of weights that are in the range of -1,1
        # this is just because the sigmoid activation produces output that is in the range of, -1,1
        # so its just a pre-optimization of sorts. 
        self.weights = (2 * np.random.random((3,1)) - 1)

    # mathematically convenient method for normalizing the output of the weighted sums
    def __sigmoid(self, x):
        return 1 / (1+np.exp(-x))
    

    # this tells us how confident we are about the output
    def __sigmoid_derivative(self, x):
        return x*(1-x)
    
    def train(self, training_set_inputs, training_set_outputs, number_of_iterations):
        for x in range(number_of_iterations):
            outputs = self.think(training_set_inputs)
            error = training_set_outputs - outputs

            adjustment = np.dot(training_set_inputs.T, error * self.__sigmoid_derivative(outputs))

            self.weights += adjustment

    def think(self, inputs):
        return self.__sigmoid(np.dot(inputs, self.weights))# should return a list of outputs for all the inputs

    




if __name__ == "__main__":
    neural_network = NeuralNetwork()

    training_set_input =  np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_output = np.array([[0, 1, 1, 0]]).T

    print("Starting weights:")
    print(neural_network.weights)


    neural_network.train(training_set_input, training_set_output, 10000)

    print ("Weights after training:")
    print(neural_network.weights)


    print("Neural network estimation of new situation: [1,0,0]")
    print(neural_network.think(np.array([1,0,0])))
