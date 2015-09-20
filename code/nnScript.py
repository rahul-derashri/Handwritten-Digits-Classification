import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time
start_time = time.time()


def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    return  1.0/(1.0+np.exp(-z))
    
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
     
    #Your code here
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
    A = mat.get('train0')
    B = mat.get('test0')
    a = range(A.shape[0])
    aperm = np.random.permutation(a)
    validation_data = np.double(A[aperm[0:1000],:])/255
    train_data = np.double(A[aperm[1000:],:])/255
    test_data = np.double(B)/255
    
    train_label = np.zeros((train_data.shape[0],10))
    train_label[:,0] = 1;
    
    validation_label = np.zeros((validation_data.shape[0],10))
    validation_label[:,0] = 1;
    
    test_label = np.zeros((test_data.shape[0],10))
    test_label[:,0] = 1;
    
    for i in range(1,10):
        A = mat.get('train'+str(i))
        B = np.double(mat.get('test'+str(i)))/255
        a = range(A.shape[0])
        aperm = np.random.permutation(a)
        A1 = np.double(A[aperm[0:1000],:])/255
        A2 = np.double(A[aperm[1000:],:])/255
        
        validation_data = np.concatenate((validation_data,A1),axis=0) 
        train_data = np.concatenate((train_data, A2),axis=0)
        
        temp_training_label = np.zeros((A2.shape[0],10))
        temp_training_label[:,i] = 1;
        train_label = np.concatenate((train_label , temp_training_label))

        temp_validation_label = np.zeros((A1.shape[0],10))
        temp_validation_label[:,i] = 1;
        validation_label = np.concatenate((validation_label , temp_validation_label))
        
        test_data = np.concatenate((test_data, B),axis=0)
        
        temp_test_label = np.zeros((B.shape[0],10))
        temp_test_label[:,i] = 1;
        test_label = np.concatenate((test_label , temp_test_label))
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
    
    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    #50X785
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    #10X51
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0  
    
    
    # ***** Step 1 - calculation for Zj (Eqn 2) *****
    temp_training_data = np.concatenate((training_data ,np.ones((training_data.shape[0], 1)) ),axis=1) 
    temp_training_data = np.transpose(temp_training_data)
    #50*50000
    Z = sigmoid(np.dot(w1 , temp_training_data))
    
    
    # ***** Step 2 - Calculation for Ol (Eqn 4) *****
    # 51X50000
    Z_Bias = np.concatenate((Z ,np.ones((1, Z.shape[1])) ),axis=0) 
    O = sigmoid(np.dot(w2 , Z_Bias))
    
    
    # ***** Step 3 - Calculation for Delta ( Eqn 9) *****
    # 10X50000
    training_label_transpose = np.transpose(training_label)
    # (Eqn 9)10X50000
    deltaL = O - training_label_transpose
    
    # Eqn 8
    #10X50
    #J = np.dot(deltaL , np.transpose(Z))
    # Eqn 16
    
    
    # ***** Step 4 - Calculation for obj_val using Eqn 5 and 15 *****
    A = -np.sum(Tlabels * np.log(Clabels) + (1-Tlabels)*np.log(1-Clabels) for (Clabels,Tlabels) in zip( O, np.transpose(training_label)))
    obj_val = np.sum(A)/training_label.shape[0]
    # Calculation for regularization value
    reg = (lambdaval * (np.sum(w1 * w1) + np.sum(w2 * w2)))/(2 * training_label.shape[0])
    # Final obj_val
    obj_val = obj_val + reg
    
    
    # ***** Step 5 - Calculation for grad_w2 ( Eqn 9 and 16 ) *****
    grad_w2 = (np.dot(deltaL , np.transpose(Z_Bias)) + lambdaval * w2)/training_label.shape[0]
    
    
    # ***** Step 6 - Calculation for grad_w1 ( Eqn 12 and 17 ) *****
    Z_Bias = np.transpose(Z)
    #51X50000
    w2 = w2[:,range(0,w2.shape[1]-1)]
    iCal_1 = ((1 - Z) * Z) * ( np.dot( np.transpose(w2), deltaL ))
    iCal_2 = np.dot(iCal_1 , np.transpose(temp_training_data))
    iCal_2 = np.transpose( iCal_2 )
    #w1 = w1[:,range(0,w1.shape[1]-1)]
    iCal_3 = lambdaval * np.transpose(w1)
    grad_w1 = (iCal_2 + iCal_3)/training_data.shape[0]
    grad_w1 = np.transpose(grad_w1)
    
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    
    return (obj_val,obj_grad)



def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    # Calculation for Z
    data = np.concatenate((data ,np.ones((data.shape[0], 1)) ),axis=1) 
    data = np.transpose(data)
    #50*50000
    Z = sigmoid(np.dot(w1 , data))
    Z_Bias = np.concatenate((Z ,np.ones((1, Z.shape[1])) ),axis=0) 
    # Calculation for Ol (10X50000)
    O = sigmoid(np.dot(w2 , Z_Bias))
    
    O = np.transpose(O)
    
    # Calculating label
    temp = O.argmax(axis=1)
    temp = np.transpose(temp)
    return temp
    



"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)
print("-----%s seconds-----"%(time.time()-start_time))

#find the accuracy on Training Dataset
# Changed basecode with TA's permission
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label.argmax(axis=1)).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset
# Changed basecode with TA's permission
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label.argmax(axis=1)).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset
# Changed basecode with TA's permission
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label.argmax(axis=1)).astype(float))) + '%')