mcculloch pits model:

inputs = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1)
]
weights = [1, 1]

threshold = 2

def mc_pitts_neuron(x1, x2, weights, threshold):

    weighted_sum = x1 * weights[0] + x2 * weights[1]
  
    if weighted_sum >= threshold:
        return 1
    else:
        return 0

print("X1 X2 | Output")
for x1, x2 in inputs:
    output = mc_pitts_neuron(x1, x2, weights, threshold)
    print(f"{x1}  {x2}  |   {output}")





single layer perceptron:

import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(input_size + 1)  
    
    def activation(self, x):
        return 1 if x >= 0 else -1
    
    def predict(self, x):
        x_with_bias = np.insert(x, 0, 1)  
        return self.activation(np.dot(self.weights, x_with_bias))
    
    def fit(self, X, y):
        for _ in range(self.epochs):
            for i in range(X.shape[0]):
                x_with_bias = np.insert(X[i], 0, 1)  
                prediction = self.predict(X[i])
                error = y[i] - prediction
                self.weights += self.learning_rate * error * x_with_bias
    
if __name__ == "__main__":

    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    y = np.array([-1, 1, 1, 1])
    
    perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=10)
    perceptron.fit(X, y)
    
    for x in X:
        print(f"Input: {x}, Predicted Output: {perceptron.predict(x)}")

    print("\nFinal Weights and Bias:")
    print(f"Weights: {perceptron.weights[1:]}")
    print(f"Bias: {perceptron.weights[0]}")






back propagation:


import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

y = np.array([[0], [1], [1], [1]])

np.random.seed(1)

input_layer_neurons = 2  
hidden_layer_neurons = 4  
output_layer_neurons = 1  

hidden_weights = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
hidden_bias = np.random.uniform(size=(1, hidden_layer_neurons))
output_weights = np.random.uniform(size=(hidden_layer_neurons, output_layer_neurons))
output_bias = np.random.uniform(size=(1, output_layer_neurons))

lr = 0.5

epochs = 5000
for epoch in range(epochs):
    hidden_layer_activation = np.dot(X, hidden_weights) + hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, output_weights) + output_bias
    predicted_output = sigmoid(output_layer_activation)

    error = y - predicted_output

    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
    hidden_weights += X.T.dot(d_hidden_layer) * lr
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

    if epoch % 1000 == 0:
        loss = np.mean(np.abs(error))
        print(f'Epoch {epoch}, Loss: {loss}')


print('\nFinal Output:')
print(predicted_output)





Genetic Algorithm:

import numpy as np
import random

POP_SIZE = 15 
GENES = 8  
MUTATION_RATE = 0.2 
GENERATIONS = 30  

def fitness(individual):
    x = int("".join(map(str, individual)), 2)  
    return x ** 2

def init_population():
    return [np.random.randint(0, 2, GENES).tolist() for _ in range(POP_SIZE)]

def select(pop):
    fitnesses = [fitness(ind) for ind in pop]
    total_fitness = sum(fitnesses)
    probabilities = [f / total_fitness for f in fitnesses]
    return pop[np.random.choice(len(pop), p=probabilities)]

def crossover(parent1, parent2):
    point = random.randint(1, GENES - 1)
    offspring1 = parent1[:point] + parent2[point:]
    offspring2 = parent2[:point] + parent1[point:]
    return offspring1, offspring2

def mutate(individual):
    for i in range(GENES):
        if random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i]
    return individual

def genetic_algorithm():
    population = init_population()
    for gen in range(GENERATIONS):
        fitness_values = [fitness(ind) for ind in population]
        best_individual = max(population, key=fitness)
        print(f"Generation {gen + 1}: Best Fitness = {fitness(best_individual)}, Best Individual = {best_individual}")
        
        if max(fitness_values) >= 255**2:  
            print("Problem Solved!")
            break
        
        new_population = []
        for _ in range(POP_SIZE // 2):
            parent1, parent2 = select(population), select(population)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])
        
        population = new_population

genetic_algorithm()





fuzzy controller:



import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl


water_level=ctrl.Antecedent(np.arange(0,11,1),'water_level')
dirt_level=ctrl.Antecedent(np.arange(0,11,1),'dirt_level')

wast_time=ctrl.Consequent(np.arange(0,61,1),'wash_time')

water_level['low']=fuzz.trimf(water_level.universe,[0,0,5])
water_level['medium']=fuzz.trimf(water_level.universe,[0,5,10])
water_level['high']=fuzz.trimf(water_level.universe,[5,10,10])

dirt_level['low']=fuzz.trimf(dirt_level.universe,[0,0,5])
dirt_level['medium']=fuzz.trimf(dirt_level.universe,[0,5,10])
dirt_level['high']=fuzz.trimf(dirt_level.universe,[5,10,10])

wash_time['short']=fuzz.trimf(wash_time.universe,[0,0,30])
wash_time['medium']=fuzz.trimf(wash_time.universe,[10,30,50])
wash_time['long']=fuzz.trimf(wash_time.universe,[30,60,60])

rule1=ctrl.Rule(water_level['low']&dirt_level['low'],wash_time['short'])
rule2=ctrl.Rule(water_level['medium']&dirt_level['medium'],wash_time['medium'])
rule3=ctrl.Rule(water_level['high']&dirt_level['high'],wash_time['long'])
rule4=ctrl.Rule(water_level['low']&dirt_level['high'],wash_time['medium'])
rule5=ctrl.Rule(water_level['high']&dirt_level['low'],wash_time['medium'])

wash_time_ctrl=ctrl.ControlSystem([rule1,rule2,rule3,rule4,rule5])
washing_machine=ctrl.ControlSystemSimulation(wash_time_ctrl)

washing_machine.input['water_level']=6
washing_machine.input['dirt_level']=7
washing_machine.compute()

print(f"Wash Time: {washing_machine.output['wash_time']:.2f} minutes")






exp 1:


import numpy as np

# Define fuzzy relation R: X × Y
R = np.array([
    [0.7, 0.6],  # μR(x1, y1), μR(x1, y2)
    [0.8, 0.3]   # μR(x2, y1), μR(x2, y2)
])

# Define fuzzy relation S: Y × Z
S = np.array([
    [0.8, 0.5, 0.4],  # μS(y1, z1), μS(y1, z2), μS(y1, z3)
    [0.1, 0.6, 0.7]   # μS(y2, z1), μS(y2, z2), μS(y2, z3)
])

# Perform Max-Min Composition
def max_min_composition(R, S):
    result = np.zeros((R.shape[0], S.shape[1]))
    
    for i in range(R.shape[0]):  # For each row in R (X)
        for j in range(S.shape[1]):  # For each column in S (Z)
            min_values = [min(R[i][k], S[k][j]) for k in range(R.shape[1])]
            result[i][j] = max(min_values)
    
    return result

T = max_min_composition(R, S)

# Print the final result
print("Result of Max-Min Composition (T = R o S):")
print(T)




import numpy as np

# Define fuzzy relation R: X × Y
R = np.array([
    [0.7, 0.6],
    [0.8, 0.3]
])

# Define fuzzy relation S: Y × Z
S = np.array([
    [0.8, 0.5, 0.4],
    [0.1, 0.6, 0.7]
])

# Perform Max-Product Composition
def max_product_composition(R, S):
    result = np.zeros((R.shape[0], S.shape[1]))

    for i in range(R.shape[0]):  # X
        for j in range(S.shape[1]):  # Z
            products = [R[i][k] * S[k][j] for k in range(R.shape[1])]
            result[i][j] = max(products)
    
    return result

T_product = max_product_composition(R, S)

# Print the result
print("Result of Max-Product Composition (T = R o S):")
print(T_product)

