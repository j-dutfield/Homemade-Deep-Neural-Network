import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')
import time
class ANN:

    #==========================================#
    # The init method is called when an object #
    # is created. It can be used to initialize #
    # the attributes of the class.             #
    #==========================================#
    def __init__(self, no_inputs=784, hidden_layer_config=[28,28,28,28], output_layer_size=10,
                max_iterations=20, learning_rate=0.005, activation = 'sigmoid', tuner='off', seed_number=2,\
                early_termination = 'off', dropout='off'):
        
        assert output_layer_size == 1 or output_layer_size == 10
        print('Initialising Neural Network')
        
        #Defining self statement
        self.seed_number=seed_number
        self.dropout=dropout
        self.early_termination = early_termination
        self.tuner = tuner
        self.no_inputs = no_inputs
        self.hidden_layer_config = hidden_layer_config
        self.output_layer_size = output_layer_size
        self.activation = activation
        layer_config = [no_inputs] + hidden_layer_config + [output_layer_size]
        self.layer_config = layer_config

        # initialise weights
        layer_weight = {}
        
        if activation == 'sigmoid':
            
            for layer in range(len(layer_config)-1):
                np.random.seed(seed_number*layer)
                node_weights = (2*np.random.random((layer_config[layer+1],layer_config[layer]))-1)\
                /layer_config[layer]
                      
                layer_weight[layer] = np.matrix(node_weights)
                  
            layer_bias = {}
            for layer in range(0,len(layer_config)-1):
                np.random.seed((seed_number+1)*layer)
                node_bias = (2*np.random.random((1,layer_config[layer+1]))-1)\
                /layer_config[layer+1]
                
                layer_bias[layer] = np.matrix(node_bias).T
                
        elif activation == 'rectifier':
            # HE initialisation
            for layer in range(len(layer_config)-1):
                np.random.seed(seed_number*layer)
                node_weights = np.multiply((np.random.randn(layer_config[layer+1],layer_config[layer]))\
                ,(np.sqrt(2/(layer_config[layer+1]+layer_config[layer]))))
                
                layer_weight[layer] = np.matrix(node_weights)
                
            layer_bias = {}
            for layer in range(0,len(layer_config)-1):
                np.random.seed((seed_number+1)*layer)
                node_bias = np.multiply((np.random.randn(1,layer_config[layer+1]))\
                ,(2/(layer_config[layer+1]+layer_config[layer])))
                
                layer_bias[layer] = np.matrix(node_bias).T
        
        self.layer_weight = layer_weight
        self.layer_bias = layer_bias
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
    
    def load_MNIST_data(self):
        
        print('Loading MNIST Datasets')
        self.training_data = np.loadtxt('mnist_train.csv', delimiter=',')
        test_data = np.loadtxt('mnist_test.csv', delimiter=',')
        
        #Splitting test data from labels    
        test_labels = []
        test_pixels = []
        for i in test_data:
            test_labels.append(int(i[0]))
            temp=i.copy()
            temp = temp[1:] #turning the first term into the bias
            test_pixels.append(temp)
        
        self.test_pixels = test_pixels
        self.test_labels = test_labels
        
    def convert_MNIST_binary(self, x):
        
        '''Converts MNIST training and test data labels into binary with\
        respect to x. Must be fun after self.load_MNIST_data function.'''
        print('Converting labels to binary format')
        
        binary_converted_data = self.training_data.copy()
        for i in binary_converted_data:
            if i[0] == x:
                i[0] = 1
            else:
                i[0] = 0
         
        x_labels = []
        for i in self.test_labels:
            if i == x:
                x_labels.append(1)
            else:
                x_labels.append(0)     
            
        self.training_data = binary_converted_data
        self.test_labels = x_labels
        self.binary_label = x
    
    def activate(self, a):
        
        if self.activation == 'sigmoid':
            activation = 1 / (1+np.exp(-a))
            
        elif self.activation == 'rectifier':
            a_shape = a.shape
            a_length = a_shape[0] * a_shape[1]
            activation = []
            
            for i in range(a_length):
                activation.append(max(0, a.item(i)))
                
            activation = np.matrix(activation)
            activation.shape = a_shape

        return activation

    def activation_prime(self, x):
        
        if self.activation == 'sigmoid':
            return np.exp(-x) / (np.square(1+np.exp(-x)))
        
        elif self.activation == 'rectifier':
            x_shape = x.shape
            x_length = x_shape[0] * x_shape[1]
            activation_prime = []
            
            for i in range(x_length):
                if x.item(i) <= 0:
                    activation_prime.append(0)
                else:
                    activation_prime.append(1)
                
            activation_prime = np.matrix(activation_prime)
            activation_prime.shape = x_shape
            return activation_prime

    def predict(self, pixels):
                # feed forward
            layer_output = {}
            layer_output[-1] = np.matrix(pixels).T
            for layer in range(len(self.layer_config)-1):
                layer_output[layer] = self.activate(np.dot(self.layer_weight[layer], layer_output[layer-1])+self.layer_bias[layer])
            
            outputs = layer_output[len(self.layer_config)-2].T
            self.layer_output = layer_output
            
            #slightly different predictor required for binary and full problems
            if self.output_layer_size == 10:
            
                outputs = outputs.tolist()
                outputs = outputs[0]
                highest_prob = outputs.index(max(outputs))
                
                rounded_outputs = []
                for i in outputs:
                    if self.activation == 'rectifier':
                        if i > 0.5:
                            rounded_outputs.append(1)
                        else:
                            rounded_outputs.append(0)
                        
                    elif self.activation == 'sigmoid':   
                        rounded_outputs.append(round(i))
                
                return rounded_outputs, outputs, highest_prob
        
            elif self.output_layer_size == 1:
                outputs = outputs.tolist()[0]
                #if self.activation == 'rectifier':
                #    if outputs[0] > 0:
                #        outputs[0] = 1
                #    return [outputs[0]], outputs, outputs[0]
                
                return [round(outputs[0])], outputs, round(outputs[0])
                
    #===============================#
    # Trains the net using labelled #
    # training data.                #
    #===============================#
    def train(self, training_data):
        
        print('Training Started')
        
        training_data = self.training_data
        index = list(range(len(training_data)-1))
        self.overfit = {}
        self.overfit[0] = 1
        max_iterations = self.max_iterations
        test_accuracy = {}
        
        training_labels = []
        training_pixels = []
        for i in training_data:
            training_labels.append(int(i[0]))
            temp=i.copy()
            temp = temp[1:] #turning the first term into the bias
            training_pixels.append(temp)
            
        assert len(training_pixels) == len(training_labels)
        
        for iter_ in range(max_iterations):
            #shuffled = call_shuffled_training(training_data)
            random.seed(self.seed_number*(iter_+1))
            random.shuffle(index)
                      
            if self.early_termination == 'on':
                if iter_ in list(range(2,self.max_iterations,2)):
                    self.test(self.test_pixels, self.test_labels, statements='off')
                    test_accuracy[iter_] = self.accuracy
                    self.test(training_pixels[:10000], training_labels[:10000], statements='off')
                    self.overfit[iter_] = self.accuracy - test_accuracy[iter_]
                    if self.overfit[iter_] > 0.08 and self.accuracy > 0.9:
                        print('Early Termination Due to Overfit')
                        self.iterations_trained = iter_
                        break
                    
                    elif iter_ >= 6 and (test_accuracy[iter_] - test_accuracy[iter_-2]) < 0.01 and test_accuracy[iter_] < 0.8:
                        print('Early Termination Due to Lack of Progress')
                        break
                    print('Overfit and convergence tests passed.')
                    
            elif self.tuner == 'on':
                if test_accuracy[iter_] > 0.82:
                    self.learning_rate = 0.01
                    self.tuner = 'second_pass'
                    print(test_accuracy)
                    print('Learning rate reduced to 0.01.')
            elif self.tuner == 'second_pass':
                if test_accuracy[iter_] > 0.89:
                    self.learning_rate = 0.005
                    print('test_accuracy')
                    print('Learning rate reduced to 0.005.')
                    self.tuner = 'off'
                    
                    
            self.iterations_trained = iter_
            for k in index:

                
                data = training_pixels[k]
                label = training_labels[k]
                
                if self.dropout == 'on':
                    
                    #copy weight dictionary
                    layer_weight_dropout = self.layer_weight
                    
                    #for single layer dropout at each iteration
                    random.seed(self.seed_number+k+iter_)
                    layer = random.randint(0,len(self.layer_config)-2)
                    
                    #for layer in range(0,len(self.layer_config)-1):
                        
                    dropout_rows = []
                    for j in range(self.layer_config[layer+1]):
                        random.seed(self.seed_number+k+iter_*(j+1))
                        
                        
                        # use line below for 0.5 dropout
                        dropout_rows.append(random.randint(0,1))
                        
                        #unhash lines below for 0.9 dropout
                        
                        #dropout_rows.append(random.randint(0,9))
                        #for k in dropout_rows:
                            #if k != 0:
                                #k=1
                        
                    dropout_matrix = np.matrix(dropout_rows).T * np.matrix(np.ones([1,self.layer_config[layer]]))
                    layer_weight_dropout[layer] = np.multiply(self.layer_weight[layer], dropout_matrix)
                        
                    # forward propogation
                    layer_output = {}
                    a = {}
                    layer_output[-1] = np.matrix(data).T
                    for layer in range(len(self.layer_config)-1):
                        a[layer] = np.dot(layer_weight_dropout[layer], layer_output[layer-1])+self.layer_bias[layer]
                        layer_output[layer] = self.activate(a[layer])
                    pass 
                        
                elif self.dropout == 'off':            
                        
                        # forward propogation
                    layer_output = {}
                    a = {}
                    layer_output[-1] = np.matrix(data).T
                    for layer in range(len(self.layer_config)-1):
                        a[layer] = np.dot(self.layer_weight[layer], layer_output[layer-1])+self.layer_bias[layer]
                        layer_output[layer] = self.activate(a[layer])
                 
                #backpropogation
                
                if self.output_layer_size == 10:
                #output_layer labels
                    output_labels = [0,0,0,0,0,0,0,0,0,0]
                    output_labels[label]=1
                    output_labels=np.matrix(output_labels)
                    
                else:
                    output_labels = np.matrix(label)
                
                
                layer_error = {}
                layer_gradient = {}
                
                #if self.activation == 'sigmoid':
                 #       layer_error[len(self.layer_config)-1] = np.multiply((layer_output[len(self.layer_config)-2].T - output_labels),self.activation_prime(layer_output[len(self.layer_config)-2].T))
                    
                #else:
                layer_error[len(self.layer_config)-1] = layer_output[len(self.layer_config)-2].T - output_labels
                    
                # hidden layer errors and gradients
                for layer in range(len(self.layer_config)-2, -1, -1):
                        
                    layer_error[layer] = np.multiply((layer_error[layer+1] * self.layer_weight[layer]), self.activation_prime(layer_output[layer-1].T))
                    
                    #layer_error[layer] = layer_error[layer+1] * self.layer_weight[layer]
                    
                    layer_gradient[layer] = layer_error[layer+1].T * layer_output[layer-1].T
    
                    #updating weights and bias
                for layer in range(len(self.layer_config)-1):
                    self.layer_weight[layer] -= self.learning_rate * layer_gradient[layer]
    
                    self.layer_bias[layer] -= self.learning_rate * layer_error[layer+1].T
                    
        #parameters to assist in debugging
        self.layer_output = layer_output      
        self.layer_error = layer_error
        self.layer_gradient = layer_gradient

    #=========================================#
    # Tests the prediction on each element of #
    # the testing data. Prints the precision, #
    # recall, and accuracy.                   #
    #=========================================#
    def test(self, testing_data, labels, statements='on'):
        assert len(testing_data) == len(labels)
        
        print('Testing Started')
        
        #initialising metrics dictionaries
        tp = {}
        fp = {}
        tn = {}
        fn = {}
        accuracy = {}
        precision = {}
        recall = {}
        f1_score = {}
        correct = 0
        incorrect = 0
        
        
        for i in range(self.output_layer_size):
            tp[i] = 0
            fp[i] = 0
            tn[i] = 0
            fn[i] = 0
        
        for data, label in zip(testing_data, labels):
            
            prediction = self.predict(data)
            
            
            output_labels = [label]
            
            # calculating metrics for individual number performance
            if self.output_layer_size == 10:
                #output_layer labels
                output_labels = [0,0,0,0,0,0,0,0,0,0]
                output_labels[label]=1
                
                #counting tp, fp, fn, fn values
            for i in range(len(prediction[0])):
                if prediction[0][i] == 1 and output_labels[i] == 1:
                    tp[i] += 1
                elif prediction[0][i] == 1 and output_labels[i] == 0:
                    fp[i] += 1
                elif prediction[0][i] == 0 and output_labels[i] == 0:
                    tn[i] += 1
                elif prediction[0][i] == 0 and output_labels[i] == 1:
                    fn[i] += 1
            
            # for overall model accuracy (based upon maximum weighted figure)
            if prediction[2] == label:
                correct+=1
            elif prediction[2] != label:
                incorrect+=1
            
            self.total_accuracy = correct / (correct+incorrect)

            if statements == 'on':
                print('Expected = %d, Predicted = %d' % (label, prediction[2]))
                print(prediction[1])
                print("Overall Accuracy:\t"+str(self.total_accuracy))
        
        if statements == 'on':
        
            for i in range(len(prediction[0])):
                accuracy[i] = (tp[i]+tn[i]) / (tp[i]+tn[i]+fp[i]+fn[i])
                precision[i] = tp[i] / (tp[i]+fp[i])
                recall[i] = tp[i] / (tp[i]+fn[i])
                f1_score[i] = 2*(precision[i]*recall[i])/(precision[i]+recall[i])          
                _index = i
                
                if self.output_layer_size == 1:
                    _index = self.binary_label
                
                print('Number %d: Accuracy=%f, Precision=%f, Recall=%f, f1_score=%f'\
                      % (_index, accuracy[i], precision[i], recall[i], f1_score[i]))
            
        self.accuracy = accuracy
        self.recall = recall
        self.precision = precision
        self.f1_score = f1_score
        
#=================================#
# Main method: executed only when #
# the program is run directly and #
# not executed when imported as a #
# module.                         #
#=================================#
if __name__ == '__main__':
    
    '''Main Method - 10-Output MNIST Classifier'''
    net = ANN()
    net.load_MNIST_data()
    net.train(net.training_data)
    net.test(net.test_pixels, net.test_labels)

def randomised_CV(iterations=12):
       
    number_of_layer_option = list(range(1,5))    
    
    learning_rate = list(np.arange(0.03,0.17,0.01))
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

#CV_results = randomised_CV()
#CV_results

