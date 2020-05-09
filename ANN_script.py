# ANN Script by James Dutfield

import numpy as np
#import sys
import random
#from os import system, name
#from time import sleep
import time
import warnings
warnings.filterwarnings('ignore')
from ANN import *


# Initialising ANN, loading data, and converted to binary classification
#problem with respect to the number 7

'''In terms of accuracy I found that the single-digit scores were slightly
better for the deep neural-net than the single-layer perceptron. The
convergence was quicker for the DNN than perceptron, with the single-output
sigmoid NN achieving 98% accuracy after a single iteration through the full
training data.

For the sigmoid function I found that it performed best with a learning rate
in the region of 0.01-0.1. When implementic the rectifier function the network
was initially suffering from the 'dying relu' problem - which was solved
by changing to an He weight initialization and dropping the learning rate by
a large amount (LR=0.0001 for single-digit model and LR=0.000005 for 10 digit
model). With a learning rate that low a lot more iterations were needed and the
 model likely got stuck in local minima, which limited it's overall accuracy
 (in the case of the 10-digit model).
 
The 95% accuracy milestone target was exceeded for the single-digit sigmoid
and relu models.

Each model in this script includes training and test stages.

For the model below, please unhash each section individually and run the script.

'''

#/////////////Model 1 - Binary DNN with Sigmoid Activation\\\\\\\\\\\\\\\\

neural_net_binary = ANN(max_iterations=1, learning_rate=0.1,\
                        hidden_layer_config=[28, 28, 28], output_layer_size=1,\
                        seed_number=2)
neural_net_binary.load_MNIST_data()
neural_net_binary.convert_MNIST_binary(7)
neural_net_binary.train(neural_net_binary.training_data)
neural_net_binary.test(neural_net_binary.test_pixels, neural_net_binary.test_labels)

score_metrics1 = [neural_net_binary.accuracy, neural_net_binary.precision,\
neural_net_binary.recall, neural_net_binary.f1_score]

#/////////////////////////// 98.0% accuracy \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#-------------------------------------------------------------------------


#/////////////Model 2 - Binary DNN with Rectifier Activation\\\\\\\\\\\\\\\\

#neural_net_binary_rect = ANN(max_iterations=10, learning_rate=0.0001,\
#                        hidden_layer_config=[28, 28, 28, 28], output_layer_size=1,\
#                        activation = 'rectifier', seed_number=9)
#neural_net_binary_rect.load_MNIST_data()
#neural_net_binary_rect.convert_MNIST_binary(7)
#neural_net_binary_rect.train(neural_net_binary_rect.training_data)
#neural_net_binary_rect.test(neural_net_binary_rect.test_pixels, neural_net_binary_rect.test_labels)
#score_metrics2 = [neural_net_binary_rect.accuracy, neural_net_binary_rect.precision,\
#neural_net_binary_rect.recall, neural_net_binary_rect.f1_score]

#/////////////////////////// 98.3% accuracy \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#-------------------------------------------------------------------------


#///////////Model 3 - Ten-output DNN with Sigmoid Activation\\\\\\\\\\\\\\

#neural_net = ANN(max_iterations=20, learning_rate=0.005,\
#                 hidden_layer_config=[28,28,28,28], output_layer_size=10,\
#                         activation = 'sigmoid', seed_number=2)
#neural_net.load_MNIST_data()
#neural_net.train(neural_net.training_data)
#neural_net.test(neural_net.test_pixels, neural_net.test_labels)

#score_metrics3 = [neural_net.accuracy, neural_net.precision,\
#neural_net.recall, neural_net.f1_score]
#-------------------------------------------------------------------------



#//////////Model 4 - Ten-output DNN with Rectifier Activation\\\\\\\\\\\\\
'''
Requires really low LR to prevent dying relu problem. Becuase of the low LR is
 requires a lot more iterations than other models.'''

#neural_net_rect = ANN(max_iterations=25, learning_rate=0.000005,\
#                 hidden_layer_config=[28,28,28,28], output_layer_size=10,\
#                         activation = 'rectifier', seed_number=12)
#neural_net_rect.load_MNIST_data()
#neural_net_rect.train(neural_net_rect.training_data)
#neural_net_rect.test(neural_net_rect.test_pixels, neural_net_rect.test_labels)

#score_metrics4 = [neural_net_rect.accuracy, neural_net_rect.precision,\
#neural_net_rect.recall, neural_net_rect.f1_score]
#-------------------------------------------------------------------------



'''Additional features of early termination and LR-tuner were developed to
speed up the randomised_CV function shown below for model optimisation. The
learning rate was found to be a more important factor than the architecture.

A dropout function was also created however this yielded no improvements so
was not included in any of the scripted models above.'''



def randomised_CV(iterations=12):
       
    number_of_layer_option = list(range(1,5))    
    
    learning_rate = list(np.arange(0.0001,0.01,0.0003))
    random.seed(time.time())
    CV_seed = random.randint(5,100)
    #CV_seed = 24
    
    ANN_dict = {}
    for i in range(iterations):
        if i > 2 and ANN_dict[i-1][0] < 0.6 and ANN_dict[i-2][0] < 0.6:
            CV_seed +=1
            
        elif i > 3 and ANN_dict[i-1][0] < 0.8 and ANN_dict[i-2][0] < 0.8 and ANN_dict[i-3][0] < 0.8:
            CV_seed +=1
            
        limit=112
        number_of_layers = np.random.choice(number_of_layer_option)
        hidden_layers = []
        for j in range(number_of_layers):
            layer_size = np.random.choice(list(range(28,limit+1,14)))
            hidden_layers.append(layer_size)
            limit=layer_size
        learning = np.random.choice(learning_rate)
        #init model
        temp_ANN= ANN(hidden_layer_config=hidden_layers,learning_rate=learning, seed_number=CV_seed)
        print(hidden_layers, learning)
        #train_model
        temp_ANN.train(training_data)
        #test_model
        temp_ANN.test(test_pixels, test_labels, statements='off')
        
        ANN_dict[i] = temp_ANN.accuracy, hidden_layers, learning, temp_ANN.iterations_trained, temp_ANN.seed_number, temp_ANN
        print(ANN_dict[i])
        print(i)
    return ANN_dict

def learning_rate_search():
     #based upon 28 28 28 model with 10 iterations 
    
    learning_rate = list(np.arange(0.05,0.16,0.05))
    count=0
    ANN_dict = {}
    for i in learning_rate:
        #init model
        temp_ANN= ANN(learning_rate=i)
        print(i)
        #traing_model
        temp_ANN.train(training_data)
        #test_model
        temp_ANN.test(test_pixels, test_labels)
        
        ANN_dict[count] = temp_ANN.accuracy, i, temp_ANN
        print(ANN_dict[count])
        count+=1
    return ANN_dict

#CV_results = randomised_CV()
#CV_results

#LR_search = learning_rate_search()
#LR_search










