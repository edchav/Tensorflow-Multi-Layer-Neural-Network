# Chavez, Edgar
# 1002_091_846
# 2023_10_15
# Assignment_02_01

import numpy as np
import tensorflow as tf

def split_data(X_train, Y_train, validation_split):
    """
    Splits the data into training and validation sets based on percentages provided.
    
    Args:
        X_train: Numpy array of input for training.
        Y_train: Numpy array of desired outputs for training samples.
        validation_split: A two-element list specifying the normalized start and end point.
 .  
    Returns:
        X_train_split: Numpy array of input for training.
        Y_train_splt: Numpy array of desired outputs for training samples.
        X_valid: Numpy array of input for validation.
        Y_valid: Numpy array of desired outputs for validation samples.
    """

    # start and end for validation sets
    num_samples = X_train.shape[0]
    start = int(num_samples * validation_split[0])
    end = int(num_samples * validation_split[1])
    
    # Initialize validation sets
    X_valid = X_train[start:end]
    Y_valid = Y_train[start:end]
    
    # Remove validation sets from training data
    X_train_split = np.concatenate([X_train[:start], X_train[end:]], axis=0)
    Y_train_split = np.concatenate([Y_train[:start], Y_train[end:]], axis=0)  

    
    return X_train_split, Y_train_split, X_valid, Y_valid

def initialize_weights(layers, input_dim, seed=2):
    """
    Initializes weights for each neural network layers, for both pre-defined weights and random weights
    
    Args:
        layers: Either a list of integers that represents number of nodes in each layer, or a list of numpy weight matrices.
        input_dim: Number of input dimensions.
        seed: Random number generator seed for initializing the weights, and for reproductility.
    
    Returns:
        weights: A list of tensorflow Variables representing the initialized weights for each layer.
    """
    weights = []
    for i in range(len(layers)):
        np.random.seed(seed)

        # If layer is an integer, initialize random weights
        if isinstance(layers[i], int):
            w = np.random.randn(input_dim+1, layers[i])
            weights.append(tf.Variable(w, dtype=tf.float32))
            input_dim = layers[i]
        
        # If layer is a numpy array, use pre-defined weights
        elif isinstance(layers[i], np.ndarray):
            weights.append(tf.Variable(layers[i], dtype=tf.float32))
            input_dim = layers[i].shape[1]-1
    return weights

def forward_propagation(inputs, weights, activations):
    """
    Performs forward propagation for each layer in the neural network.

    Args:
        inputs: Tensor input data.
        weights: A list of tensorflow Variables, representing the weights for each layer.
        activations: A list of activation functions corresponding to each layer.
    
    Returns:
        activation_outputs: A list of tensors representing the output of each layer after activation.
    """
    activation_outputs = [inputs] #List storing output of each layer after activation
    for i, w in enumerate(weights):
        # Add bias term to inputs
        inputs = tf.concat([tf.ones([tf.shape(activation_outputs[-1])[0], 1]), activation_outputs[-1]], axis=1)
        z = tf.matmul(inputs, w)

        # Activation functions
        activation_functions = {
            "linear": lambda z: z,
            "sigmoid": tf.sigmoid,
            "relu": tf.nn.relu
        }
        activated = activation_functions[activations[i].lower()](z)
        activation_outputs.append(activated)
        inputs = activated

    return activation_outputs

def compute_loss(Y_true, Y_pred, loss="mse"):
    """
    Computes the loss between actual values and predictions.

    Args:
        Y_true: Tensor of actual values.
        Y_pred: Tensor of predicted values.
        loss: A string representing the loss function to use.
    
    Returns:
        current_loss: A tensor representing the loss between actual values and predictions.
    """
    # Loss functions
    loss_function = {
        "mse": lambda Y_true, Y_pred: tf.reduce_mean(tf.square(Y_true - Y_pred)),
        "cross_entropy": lambda Y_true, Y_pred: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred, labels=Y_true)),
        "svm": lambda Y_true, Y_pred: tf.reduce_mean(tf.maximum(0., 1 - Y_true * Y_pred))
    }
    current_loss = loss_function[loss.lower()](Y_true, Y_pred)  
    return current_loss

def multi_layer_nn_tensorflow(X_train,Y_train,layers,activations,alpha,batch_size,epochs=1,loss="svm",
                              validation_split=[0.8,1.0],seed=2):
    """
    Creates and trains a multi-layer neural network using tensorflow.
    
    Args:
        X_train: numpy array of input for training [nof_train_samples,input_dimensions]
        Y_train: numpy array of desired outputs for training samples [nof_train_samples,output_dimensions]
        layers: Either a list of integers or alist of numpy weight matrices.
                If layers is a list of integers then it represents number of nodes in each layer. In this case
                the weight matrices should be initialized by random numbers.
                If the layers is given as a list of weight matrices, then the given matrices should be used and NO random
                initialization is needed.
        activations: list of case-insensitive activations strings corresponding to each layer. The possible activations
                     are, "linear", "sigmoid", "relu".
        alpha: learning rate
        epochs: number of epochs for training.
        loss: is a case-insensitive string determining the loss function. The possible inputs are: "svm" , "mse",
              "cross_entropy". for cross entropy use the tf.nn.softmax_cross_entropy_with_logits().
        validation_split: a two-element list specifying the normalized start and end point to
                          extract validation set. Use floor in case of non integers.
        seed: random number generator seed for initializing the weights.
    
    Returns:
        weights: list of weight matrices corresponding to each layer including biases
        erorr: list of errors after each epoch
        validation: numpy array of actual output of the network when validation set is used as input"""
    
    # Splits data into training and validation sets
    X_train_split, Y_train_split, X_valid, Y_valid = split_data(X_train, Y_train, validation_split)
    
    # Initialize weights
    input_dim=X_train.shape[1]
    weights = initialize_weights(layers, input_dim , seed)
    
    #training loop
    error = []
    for epoch in range(epochs):
        # Loop through each batch for training
        for batch_idx in range(0, X_train_split.shape[0], batch_size):
            X_batch = X_train_split[batch_idx:batch_idx+batch_size]
            Y_batch = Y_train_split[batch_idx:batch_idx+batch_size]

            #GradientTape for automatic differentiation
            with tf.GradientTape() as tape:
                Y_pred = forward_propagation(X_batch, weights, activations)[-1]
                current_loss = compute_loss(Y_batch, Y_pred, loss)

            #computes gradients for each weight
            gradients = tape.gradient(current_loss, weights)
            for w, grad in zip(weights, gradients):
                w.assign_sub(alpha * grad)

        #error after each epoch
        Y_pred_valid = forward_propagation(X_valid, weights, activations)[-1]   
        validation_loss = compute_loss(Y_valid, Y_pred_valid, loss)  
        error.append(validation_loss.numpy())
    

    validation = forward_propagation(X_valid, weights, activations)[-1].numpy()   

    return weights, error, validation

    # This function creates and trains a multi-layer neural Network
    # X_train: numpy array of input for training [nof_train_samples,input_dimensions]
    # Y_train: numpy array of desired outputs for training samples [nof_train_samples,output_dimensions]
    # layers: Either a list of integers or alist of numpy weight matrices.
    # If layers is a list of integers then it represents number of nodes in each layer. In this case
    # the weight matrices should be initialized by random numbers.
    # If the layers is given as a list of weight matrices, then the given matrices should be used and NO random
    # initialization is needed.
    # activations: list of case-insensitive activations strings corresponding to each layer. The possible activations
    # are, "linear", "sigmoid", "relu".
    # alpha: learning rate
    # epochs: number of epochs for training.
    # loss: is a case-insensitive string determining the loss function. The possible inputs are: "svm" , "mse",
    # "cross_entropy". for cross entropy use the tf.nn.softmax_cross_entropy_with_logits().
    # validation_split: a two-element list specifying the normalized start and end point to
    # extract validation set. Use floor in case of non integers.
    # seed: random number generator seed for initializing the weights.
    # return: This function should return a list containing 3 elements:
        # The first element of the return list should be a list of weight matrices.
        # Each element of the list should be a 2-d numpy array which corresponds to the weight matrix of the
        # corresponding layer (Biases should be included in each weight matrix in the first row).

        # The second element should be a one dimensional list of numbers
        # representing the error after each epoch. Each error should
        # be calculated by using the validation set while the network is frozen.
        # Frozen means that the weights should not be adjusted while calculating the error.

        # The third element should be a two-dimensional numpy array [nof_validation_samples,output_dimensions]
        # representing the actual output of the network when validation set is used as input.

    # Notes:
    # The data set in this assignment is the transpose of the data set in assignment_01. i.e., each row represents
    # one data sample.
    # The weights in this assignment are the transpose of the weights in assignment_01.
    # Each output weights in this assignment is the transpose of the output weights in assignment_01
    # DO NOT use any other package other than tensorflow and numpy
    # Bias should be included in the weight matrix in the first row.
    # Use steepest descent for adjusting the weights
    # Use minibatch to calculate error and adjusting the weights
    # Reseed the random number generator when initializing weights for each layer.
    # Use numpy for weight to initialize weights. Do not use tensorflow weight initialization.
    # Do not use any random method from tensorflow
    # Do not shuffle data
    # i.e., Initialize the weights for each layer by:
    # np.random.seed(seed)
    # np.random.randn()
