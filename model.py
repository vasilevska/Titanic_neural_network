import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(Z):
	A = 1./(1+np.exp(-Z))
	return A, Z

def relu(Z):
	A = np.maximum(0,Z)
	assert(A.shape == Z.shape)
	return A, Z

def tanh(Z):
	A = np.tanh(Z)
	assert(A.shape == Z.shape)
	return A, Z

def sigmoid_backward(dA, cache):
	s = 1/(1+np.exp(-cache))
	dZ = dA * s * (1 - s)
	assert(dZ.shape == cache.shape)
	return dZ

def relu_backward(dA, cache):
	dZ = np.array(dA)
	assert(dZ.shape == cache.shape)
	dZ[cache<=0] = 0
	return dZ

def tanh_backward(dA, cache):
	dZ = 1 - np.square(np.tanh(cache))
	assert(dZ.shape == cache.shape)
	return dZ

def initialize_parameters_deep(layers_sizes):

	rng = np.random.default_rng()
	parameters = {}
	L = len(layers_sizes)

	for l in range(1,L):
		parameters['W' + str(l)] = rng.standard_normal((layers_sizes[l], layers_sizes[l-1]))/np.sqrt(layers_sizes[l-1])
		parameters['b' + str(l)] = np.zeros((layers_sizes[l], 1))

		assert(parameters['W' + str(l)].shape == (layers_sizes[l], layers_sizes[l-1]))
		assert(parameters['b' + str(l)].shape == (layers_sizes[l], 1))

	return parameters

def linear_forward(A, W, b):
	Z = np.dot(W,A) + b
	assert(Z.shape == (W.shape[0], A.shape[1]))
	cache = (A, W, b)
	return Z,cache

def linear_activation_forward(A_prev, W, b, activation):
	Z, linear_cache = linear_forward(A_prev, W, b)
	if activation == "sigmoid":
		A, activation_cache = sigmoid(Z)
	elif activation == "relu":
		A, activation_cache = relu(Z)
	elif activation == "tanh":
		A, activation_cache = tanh(Z)
	assert(A.shape == (W.shape[0], A_prev.shape[1]))
	cache = (linear_cache, activation_cache)

	return A, cache

def L_model_forward(X, parameters, activation):
	caches = []
	A = X
	L = len(parameters)//2

	for l in range(1,L):
		A_prev = A
		A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], activation = "relu")
		caches.append(cache)

	AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], activation)
	caches.append(cache)

	assert(AL.shape == (1, X.shape[1]))

	return AL, caches

def compute_cost(AL, Y):
	m = Y.shape[1]

	cost = (-1. / m) * np.nansum(np.multiply(Y, np.log(AL).T) + np.multiply(1 - Y, np.log(1 - AL).T))
	cost = np.squeeze(cost)
	assert(cost.shape==())

	return cost

def linear_backwards(dZ, cache):
	A_prev, W, b = cache
	m = A_prev.shape[1]

	dW = 1./m * np.dot(dZ, A_prev.T)
	db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
	dA_prev = np.dot(W.T, dZ)

	assert(dA_prev.shape == A_prev.shape)
	assert(dW.shape == W.shape)
	assert(db.shape == b.shape)

	return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
	linear_cache, activation_cache = cache
	if activation == "relu":
		dZ = relu_backward(dA, activation_cache)
	elif activation == "sigmoid":
		dZ = sigmoid_backward(dA, activation_cache)
	elif activation == "tanh":
		dZ = tanh_backward(dA, activation_cache)

	dA_prev, dW, db = linear_backwards(dZ, linear_cache)
	return dA_prev, dW, db

def L_model_backward(AL, Y, caches, activation):
	grads = {}
	L = len(caches)
	m = AL.shape[1]
	Y = Y.reshape(AL.shape)

	dAL = -(np.divide(Y,AL) - np.divide(1 - Y, 1-AL))
	current_cache = caches[L-1]
	grads["dA"+str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation)

	for l in reversed(range(L-1)):
		current_cache = caches[l]
		dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+2)],current_cache, "relu")
		grads["dA" + str(l+1)] = dA_prev_temp
		grads["dW" + str(l+1)] = dW_temp
		grads["db" + str(l+1)] = db_temp

	return grads

def update_parameters(parameters, grads, learning_rate):
	L = len(parameters)//2

	for l in range(L):
		parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
		parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

	return parameters

def L_layer_model(X, Y, layers_sizes, activation, learning_rate = 0.005, num_iterations = 500, print_cost = False):
	costs = []
	parameters = initialize_parameters_deep(layers_sizes)
	for i in range(0, num_iterations):
		AL, caches = L_model_forward(X, parameters, activation)
		cost = compute_cost(AL,Y)
		grads = L_model_backward(AL, Y, caches, activation)
		parameters = update_parameters(parameters, grads, learning_rate)

		if print_cost and i%500 == 0:
			print("Cost after it %i: %f" %(i, cost))

		if print_cost and i%100 == 0:
			costs.append(cost)
	if print_cost:
		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per hundreds)')
		plt.title("Learning rate =" + str(learning_rate))
		plt.show()

	return parameters

def predict(X, y, parameters, activation):
	m = X.shape[1]
	n = len(parameters) // 2
	p = np.zeros((1,m))
	probs, caches = L_model_forward(X, parameters, activation)

	if activation == "sigmoid":
		for i in range(0, probs.shape[1]):
			if probs[0,i] > 0.5:
				p[0,i] = 1
			else:
				p[0,i] = 0
	elif activation == "tanh":
		for i in range(0, probs.shape[1]):
			if probs[0,i] > 0:
					p[0,i] = 1
			else:
					p[0,i] = 0

	print("Tacnost: " + str(np.sum((p == y)/m)))

	return p

def generate_csv(X, y, parameters, test, activation):
	m = X.shape[1]
	n = len(parameters) // 2
	p = np.zeros((1,m))
	probs, caches = L_model_forward(X, parameters, activation)
	if activation == "sigmoid":
		for i in range(0, probs.shape[1]):
			if probs[0,i] > 0.5:
				p[0,i] = 1
			else:
				p[0,i] = 0
	elif activation == "tanh":
		for i in range(0, probs.shape[1]):
			if probs[0,i] > 0:
					p[0,i] = 1
			else:
					p[0,i] = 0

	row = test.loc[test['PassengerId'].notnull(), 'PassengerId']
	row = row.reset_index(drop= True)
	pIds = pd.DataFrame(data=row.astype(int), columns=['PassengerId'], dtype=int)
	preds = pd.DataFrame(data=p.T,columns=['Survived'], dtype=int)
	final_prediction = pd.concat([pIds, preds], axis=1)
	final_prediction.to_csv('titanic_survival_predictions.csv', index=False)
	
	return

