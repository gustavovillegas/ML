#Importing libraries 
from keras.models import Sequential
from keras import optimizers
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import ModelCheckpoint
from keras.initializers import RandomNormal
from keras import regularizers
from keras.constraints import maxnorm
from keras.layers.normalization import BatchNormalization
from helper import PlotLosses
from keras import backend as K


#Creating new categorical_crossentropy to consider class weights
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss


#Creating models to be used on the tickets classification
#Logistic regression with 1 dense layer
def model_1(input_shape, lr):
    model = Sequential()
    model.add(Dense(4, activation='softmax',  kernel_regularizer=regularizers.l2(0), input_dim=input_shape))
    selectedOptimizer = optimizers.adam(lr=lr)
    model.compile(loss = 'categorical_crossentropy', optimizer = selectedOptimizer, metrics=['accuracy'])
    model.summary()
    return model

#Logistic regression with 3 dense layer
def model_2(input_shape, lr, dropout_rate=0.25):
    model = Sequential()
    model.add(Dense(32, activation='softmax',  kernel_regularizer=regularizers.l2(0), input_dim=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='softmax', kernel_regularizer=regularizers.l2(0), name='middle'))
    model.add(Dense(4, kernel_initializer='zeros', name='Salida'))
    model.add(Activation('softmax'))
    selectedOptimizer = optimizers.adam(lr=lr)
    model.compile(loss = 'categorical_crossentropy', optimizer = selectedOptimizer, metrics=['accuracy'])
    model.summary()
    return model

#Logistic regression with 4 dense layer
def model_3(input_shape, lr, dropout_rate=0.25):
    model = Sequential()
    model.add(Dense(32, activation='softmax',  kernel_regularizer=regularizers.l2(0), input_dim=input_shape))
    #model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='softmax', kernel_regularizer=regularizers.l2(0), name='middle'))
    #model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    #model.add(Dense(4, activation='softmax', kernel_initializer='zeros', name='middle2'))
    #model.add(BatchNormalization())
    #model.add(Dropout(dropout_rate))
    model.add(Dense(4, kernel_initializer='zeros', name='Salida'))
    model.add(Activation('softmax'))
    selectedOptimizer = optimizers.adam(lr=lr)
    model.compile(loss = 'binary_crossentropy', optimizer = selectedOptimizer, metrics=['accuracy'])
    model.summary()
    return model


#4 dense layer using a custom loss fuction to consider weights
def model_4(input_shape, lr, dropout_rate, class_weights):
    loss = weighted_categorical_crossentropy(class_weights)
    
    model = Sequential()
    model.add(Dense(32, activation='softmax',  kernel_regularizer=regularizers.l2(0), input_dim=input_shape))
    #model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='softmax', kernel_regularizer=regularizers.l2(0), name='middle'))
    #model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    #model.add(Dense(4, activation='softmax', kernel_initializer='zeros', name='middle2'))
    #model.add(BatchNormalization())
    #model.add(Dropout(dropout_rate))
    model.add(Dense(4, kernel_initializer='zeros', name='Salida'))
    model.add(Activation('softmax'))
    
    selectedOptimizer = optimizers.adam(lr=lr)
    model.compile(loss = loss, optimizer = selectedOptimizer, metrics=['accuracy'])
    model.summary()
    return model

#5 dense layer using a custom loss fuction to consider weights
def model_5(input_shape, lr, dropout_rate, class_weights):
    loss = weighted_categorical_crossentropy(class_weights)
    
    model = Sequential()
    model.add(Dense(128, activation='softmax',  kernel_regularizer=regularizers.l2(0), input_dim=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation='softmax', kernel_regularizer=regularizers.l2(0), name='middle'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(4, kernel_initializer='zeros', name='Salida'))
    model.add(Activation('softmax'))
    
    selectedOptimizer = optimizers.adam(lr=lr)
    model.compile(loss = loss, optimizer = selectedOptimizer, metrics=['accuracy'])
    model.summary()
    return model
