import numpy as np
import matplotlib.pyplot as plt
import math

# Initialize parameters using HE initialization
def params_init(layers):
    params = {}
    L = len(layers) 
    for l in range(L-1):
        params['W'+str(l+1)] = (np.random.randn(layers[l+1],layers[l])
                                *np.sqrt(2/layers[l]))
        
        params['b'+str(l+1)] = np.zeros((layers[l+1],1))
        
        assert(params['W' + str(l+1)].shape == (layers[l+1], layers[l]))
        assert(params['b' + str(l+1)].shape == (layers[l+1], 1))
        
    return params

# ReLU function
def relu(Z):

    A = np.maximum(0,Z)
    if type(Z) is np.ndarray:
        assert(A.shape == Z.shape)
    
    return A 

 # ReLu Gradient
def relu_grad(Z):
    grad = np.zeros(Z.shape)
    grad[Z>0] =1 
    if type(Z) is np.ndarray:
        assert(grad.shape == Z.shape)
    
    return grad

def softmax(Z):
    A = np.exp(Z)/(np.sum(np.exp(Z),axis = 0))
    assert(A.shape == Z.shape)
    
    return A

# Sofmax with better stability
def norm_softmax(Z):
    b = Z.max(axis=0)
    y = np.exp(Z-b)
    A = y/(np.sum(y,axis = 0))
    assert(A.shape == Z.shape)
    
    return A

# Forward Propagation
# (L-1) ReLu layers with Softmax output layer
def forward_prop(X,params):
    caches = {}
    A = X
    caches['A'+str(0)] = X
    L = len(params) // 2  # number of layers in the network
    for l in range(1,L):
        A_prev = A
        Z = np.dot(params['W'+str(l)],A_prev) + params['b'+str(l)]
        assert(Z.shape == (params['W'+str(l)].shape[0],A.shape[1]))
        caches['Z'+str(l)] = Z 
        A = relu(Z)
        caches['A'+str(l)] = A

    # Output Layer
    Z = np.dot(params['W'+str(L)],A) + params['b'+str(L)]
    assert(Z.shape == (params['W'+str(L)].shape[0],A.shape[1]))
    caches['Z'+str(L)] = Z 
    AL = norm_softmax(Z)
    
    return AL,caches

# Cost Function
def costFunction(AL,Y,reg=0):
    
    cost = np.mean(-np.sum(Y*np.log(AL),axis=0))
    cost += reg/AL.shape[1]
    
    return cost

# Perform the actual backprop
def compute_grads(dZ,grads,params,caches,l,m,lamb):
    
    dW = (1/m)* np.dot(dZ,caches['A'+str(l-1)].T) + (lamb/m)*params['W'+str(l)]
    assert (dW.shape == params['W'+str(l)].shape)
    grads['dW'+str(l)] = dW
    db = (1/m)* np.sum(dZ,axis=1,keepdims=True)
    assert (db.shape == params['b'+str(l)].shape)
    grads['db'+str(l)] = db
    dA_prev = np.dot(params['W'+str(l)].T,dZ)
    assert(dA_prev.shape == caches['A'+str(l-1)].shape)
    grads['dA'+str(l-1)] = dA_prev
    
    return grads

# Backpropagation
def backprop(AL,Y,caches,params,lamb=0):
    grads = {}
    L = len(params) //2
    m = AL.shape[1]
    # Backprop of the first layer
    dZ = AL-Y
    grads = compute_grads(dZ,grads,params,caches,L,m,lamb)
    
    # Backprop of other layers
    for l in reversed(range(1,L)):
        
        dA = grads['dA'+str(l)]
        dZ = dA * relu_grad(caches['Z'+str(l)])     
        assert(dZ.shape == caches['Z'+str(l)].shape)
        grads = compute_grads(dZ,grads,params,caches,l,m,lamb) 
        
    del grads['dA0']    
    return grads

# Converting a dictionary to vector
def dict_to_vector(params,grads):
    
    total = 0
    L = len(params) // 2
    theta = theta_grads = np.empty(0)
    for l in range(1,L+1):
        total += (params['W'+str(l)].size + params['b'+str(l)].size)
        theta = np.append(theta,params['W'+str(l)])
        theta = np.append(theta,params['b'+str(l)])
        
        theta_grads = np.append(theta_grads,grads['dW'+str(l)])
        theta_grads = np.append(theta_grads,grads['db'+str(l)])
    
    assert(total == theta.size)
    return theta.reshape(-1,1),theta_grads.reshape(-1,1)

# Converting a vector to dictionary 
def vector_to_dict(theta,p):
    
    L = len(p) // 2
    params = {}
    pos = 0 
    for l in range(1,L+1):
        w_size = p['W'+str(l)].size
        b_size = p['b'+str(l)].size
        params['W'+str(l)] = theta[pos:pos+w_size].reshape(p['W'+str(l)].shape)
        pos += w_size
        params['b'+str(l)] = theta[pos:pos+b_size].reshape(p['b'+str(l)].shape)
        pos +=b_size
        assert(params['W'+str(l)].shape == p['W'+str(l)].shape)
        assert(params['b'+str(l)].shape == p['b'+str(l)].shape)
    
    return params

# Perform gradient checking using Numerical Gradient Estimation  
# to check our implementation of backprop
def gradient_check(X,Y,params,grads):
    
    epsilon = 1e-7
    L = len(params) // 2
    theta,theta_grads = dict_to_vector(params,grads)
    num_params = theta.size
    J_plus = np.zeros((num_params,1))
    J_minus = np.zeros((num_params,1))
    grad_approx = np.zeros((num_params,1))
    
    for i in range(num_params):
        
        thetaplus = np.copy(theta)
        thetaplus[i] += epsilon
        thetaplus_dict = vector_to_dict(thetaplus,params)
        ALplus, _ = forward_prop(X,thetaplus_dict)
        J_plus[i] = costFunction(ALplus,Y)
    
        thetaminus = np.copy(theta)
        thetaminus[i] -= epsilon
        thetaminus_dict = vector_to_dict(thetaminus,params)
        ALminus, _ = forward_prop(X,thetaminus_dict)
        J_minus[i] = costFunction(ALminus,Y)
        
        grad_approx[i] = (J_plus[i] - J_minus[i])/(2*epsilon)
    
    numerator = np.linalg.norm(theta_grads-grad_approx)                              
    denominator = np.linalg.norm(theta_grads) + np.linalg.norm(grad_approx)                         
    difference = numerator/denominator 
    
    if difference > 2e-6:
        print ("\033[91m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

        
# Function to create mini batches
def random_mini_batches(X,Y,batch_size = 64):
    
    mini_batches = []
    m = X.shape[1]
    # Shuffle
    permute = np.random.permutation(m)
    X = X[:,permute]
    Y = Y[:,permute]
    
    # Partition
    complete_batches = math.floor(m/batch_size)
    for k in range(complete_batches):
        batch_X = X[:,k*batch_size:(k+1)*batch_size]
        batch_Y = Y[:,k*batch_size:(k+1)*batch_size]
        batch = (batch_X,batch_Y)
        mini_batches.append(batch)
        
    # Handling the end case
    if m % batch_size != 0:
        batch_X = X[:,(k+1)*batch_size:]
        batch_Y = Y[:,(k+1)*batch_size:]
        batch = (batch_X,batch_Y)
        mini_batches.append(batch)
        
    return mini_batches        


# Initialization for Adam Optimization
def init_for_adam(params):
    
    L = len(params) //2
    v = {}
    s = {}
    
    for l in range(L):
        
        v["dW" + str(l+1)] = np.zeros(params['W'+str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(params['b'+str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(params['W'+str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(params['b'+str(l+1)].shape)
        
    return v, s

# Adam optimization algorithm for updating weights
def update_parameters_with_adam(params, grads, v, s, t, alpha = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
   
    
    L = len(params) // 2                 
    v_corrected = {}                         
    s_corrected = {}                         
    
    for l in range(L):
        
        v["dW" + str(l+1)] = beta1*v["dW" + str(l+1)] + (1-beta1)*grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta1*v["db" + str(l+1)] + (1-beta1)*grads["db" + str(l+1)]
       
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1-beta1**t)
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1-beta1**t)
        
        s["dW" + str(l+1)] = beta2*s["dW" + str(l+1)] + (1-beta2)*(grads["dW" + str(l+1)]**2 )
        s["db" + str(l+1)] = beta2*s["db" + str(l+1)] + (1-beta2)*(grads["db" + str(l+1)]**2 )
       
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1-beta2**t)
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1-beta2**t)
       
        params["W" + str(l+1)] = params["W" + str(l+1)] - (alpha*(v_corrected["dW" + str(l+1)])/
                                      (np.sqrt(s_corrected["dW" + str(l+1)])+epsilon))
        params["b" + str(l+1)] = params["b" + str(l+1)] - (alpha*(v_corrected["db" + str(l+1)])/
                                      (np.sqrt(s_corrected["db" + str(l+1)])+epsilon))
       

    return params, v, s

# Generates a list of exponentialy increasing learning rate required by the lrFinder 
def lrList(lr=1e-6,scale=2):
    rate_list = [lr]
    while(lr<0.5):
        lr = lr*scale
        rate_list.append(lr)
    
    return rate_list  

# Learning Rate decay
def learningRate_decay(alpha,epoch_num):
    a = alpha*(0.8 ** (epoch_num/30))
    return a

# Function to compute the l2 regularization of weights (only W's)
def l2regularization(params,lamb):
    
    L = len(params) // 2
    total = 0 
    
    for l in range(L):
        total += np.sum(params['W'+str(l+1)]**2)
    
    reg_term = (lamb/2)*total
    return reg_term

# Creating a One Hot Matrix
def one_hot(y):
    C = np.unique(y).size
    y_hot = np.eye(C)[:,y.reshape(-1)]
    
    return y_hot

# Predicting Multiclass labels
def predict_multiClass(X2,params2):
    AL,_ = forward_prop(X2,params2)
    pred = np.argmax(AL,axis=0)
    return pred

#Preding the accuracy
def accuracy(Xnorm,Y,test_Xnorm,testY,p):
    pred_y = predict_multiClass(Xnorm,p)
    acc_train = np.mean(pred_y.flatten()==Y.flatten())*100
    print('Accuracy on the Training Set: %s %%' %round(acc_train,2))

    pred_ytest = predict_multiClass(test_Xnorm,p)
    acc_test = np.mean(pred_ytest.flatten()==testY.flatten())*100
    print('Accuracy on the Test Set: %s %%' %round(acc_test,2))
    print ("Neural Network made errors in predicting %s samples out of 10000 in the Test Set " 
           % np.count_nonzero(testY != pred_ytest))

# Function for computing the confusion matrix
def confusion_matrix(test_y,pred_y,labels):
    c_mat = np.zeros((labels.size,labels.size),dtype=int)
    
    for i in labels:
        pos = test_y == i
        for j in labels:
            temp = np.count_nonzero(pred_y[pos] == j)
            c_mat[i,j] = temp
    
    return c_mat   

# Plotting the confusion matrix
def plot_confusionMatrix(c_mat,class_labels):
    plt.figure(figsize=(11,9))
    ax = plt.axes()
    plt.imshow(c_mat,cmap=plt.cm.Spectral_r)
    plt.colorbar()
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted Labels",fontsize=12)
    ax.set_ylabel("True Labels",fontsize = 12)

    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            text = ax.text(j, i, c_mat[i, j],ha="center", va="center", color="w")
        
    plt.title("Confusion Matrix",fontsize=15)
    plt.show()
