import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Dense, Input, concatenate
#cifar10
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

#load data
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
#normalize
x_train=x_train/255
x_test=x_test/255
#one-hot
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)

def Inception_block(X,filter_list):
    filter_size=filter_list
    #1x1 conlutions
    con_1=Conv2D(filter_size[0],(1,1),padding='same',activation='relu')(X)
    #3x3 convolutions
    Conv_3_reduce=Conv2D(filter_size[1],(1,1),padding='same',activation='relu')(X)
    conv_3=Conv2D(filter_size[2],(3,3),padding='same',activation='relu')(Conv_3_reduce)
    #5x5 convolutions
    Conv_5_reduce=Conv2D(filter_size[3],(1,1),padding='same',activation='relu')(X)
    conv_5=Conv2D(filter_size[4],(5,5),padding='same',activation='relu')(Conv_5_reduce)
    #3x3 maxpooling
    MaxPool3=MaxPooling2D((3,3),padding='same',strides=(1,1))(X)
    Conv1=Conv2D(filter_size[5],(1,1),padding='same',activation='relu')(MaxPool3)

    Inception_output=concatenate([con_1,conv_3,conv_5,Conv1],axis=3)
    return Inception_output

def Inception_model(input_shape=(224,224,3),classes=10):
    X_input=Input(input_shape)
    x=Conv2D(64,(7,7),strides=(2,2),padding='valid',activation='relu')(X_input)
    x=MaxPooling2D((3,3),strides=(2,2),padding='same')(x)
    x=Conv2D(192,(3,3),strides=1,padding='same',activation='relu')(x)
    x=MaxPooling2D((3,3),strides=(2,2),padding='same')(x)
    x=Inception_block(x,[64,96,128,16,32,32])
    x=Inception_block(x,[128,128,192,32,96,64])
    x=MaxPooling2D((3,3),strides=(2,2),padding='same')(x)
    x=Inception_block(x,[192,96,208,16,48,64])
    x=Inception_block(x,[160,112,224,24,64,64])
    x=Inception_block(x,[128,128,256,24,64,64])
    x=Inception_block(x,[112,144,288,32,64,64])
    x=Inception_block(x,[256,160,320,32,128,128])
    x=MaxPooling2D((3,3),strides=(2,2),padding='same')(x)
    x=Inception_block(x,[256,160,320,32,128,128])
    x=Inception_block(x,[384,192,384,48,128,128])
    x=AveragePooling2D((7,7),strides=(1,1),padding='same')(x)
    x=Dropout(0.4)(x)
    x=Flatten()(x)
    x=Dense(classes,activation='softmax')(x)
    
    # Create a Keras Model connecting the input tensor to the output tensor
    model = tf.keras.models.Model(inputs=X_input, outputs=x)
    
    return model

# Create the Inception model
model = Inception_model(input_shape=(32,32,3),classes=10)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10,batch_size=128)
evaluation=model.evaluate(x_test,y_test)
accuracy = evaluation[1]  
# You can also print the evaluation results
print("Evaluation Results:")
print(f"Loss: {evaluation[0]}")
print(f"Accuracy: {accuracy}")
