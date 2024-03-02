import numpy as np
import tensorflow
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.applications import  VGG16, ResNet50
from keras.layers import Conv2D,Flatten,Dense,MaxPool2D,BatchNormalization,Dropout

class  ImgClassifier:
    
    def Preprocessing(self,img_path,shear_range=0.2,zoom_range=0.2,width_shift_range=0.2,height_shift_range=0.2,target_size=(150,150)):
        """
        Preprocess the data
        using Image Data Generator
        from Keras library

        """
        dataset = ImageDataGenerator(
            rescale= 1./255,
            shear_range=shear_range,
            horizontal_flip=True,
            zoom_range=zoom_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range
        )

        dataset_generator = dataset.flow_from_directory(
            img_path,
            target_size=target_size,
            batch_size=32,
            class_mode='categorical'
        )

        return dataset_generator
    

    def PreTrained_model(self,Neuron_conv=64,mdl=None,dense_Neuron=64,dropout=0.2,output=2):

        """
        parameters:
        dataset_generated-> the generated dataset for training
        Neuron_conv -> convolutional layer  neurons count
        mdl -> select model
        dense_neuron-> which is custom for 64
        """

        if mdl == 'ResNet50':
            print("ResNet 50")
            res_model = ResNet50(include_top=False, weights="imagenet",input_shape=(150,150,3))

            res_model.trainable = False

            model = Sequential()

            model.add(res_model)
            model.add(Flatten())

            model.add(Dense(dense_Neuron,activation='relu'))
            model.add(Dropout(dropout))

            model.add(Dense(dense_Neuron,activation='relu'))
            model.add(Dropout(dropout))

            model.add(Dense(output,activation='softmax'))

            model.summary()

            return model
        
        if mdl == 'VGG16':
            print("VGG 16 Model")
            res_model = VGG16(include_top=False, weights="imagenet",input_shape=(150,150,3))

            res_model.trainable = False

            model = Sequential()

            model.add(res_model)
            model.add(Flatten())

            model.add(Dense(dense_Neuron,activation='relu'))
            model.add(Dropout(dropout))

            model.add(Dense(dense_Neuron,activation='relu'))
            model.add(Dropout(dropout))

            model.add(Dense(output,activation='softmax'))

            model.summary()

            return model
        
        if mdl == None:
            print("Custum model Model")

            model = Sequential()

            model.add(Conv2D(Neuron_conv,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(150,150,3)))
            model.add(BatchNormalization())
            model.add(MaxPool2D(pool_size=(2, 2),strides=2,padding='valid'))

            model.add(Conv2D(Neuron_conv,kernel_size=(3,3),padding='valid',activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPool2D(pool_size=(2, 2),strides=2,padding='valid'))

            model.add(Conv2D(Neuron_conv,kernel_size=(3,3),padding='valid',activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPool2D(pool_size=(2, 2),strides=2,padding='valid'))

            model.add(Flatten())

            model.add(Dense(dense_Neuron,activation='relu'))
            model.add(Dropout(dropout))

            model.add(Dense(dense_Neuron,activation='relu'))
            model.add(Dropout(dropout))

            model.add(Dense(output,activation='softmax'))

            model.summary()

            return model
        


    def compile_and_fit_model(self,model, dataset_generated = None, epochs=5, steps_per_epoch=None):
        """
        Compile and fit the given Keras model using the specified generators.
        Parameters:
        - model: Keras model to be compiled and fit.
        - train_generator: Generator for training data.
        - test_generator: Generator for testing data.
        - epochs (int): Number of epochs for training. Default is 10.
        - steps_per_epoch (int): Number of steps (batches) to be processed in each epoch. Default is None.
        - validation_steps (int): Number of steps (batches) to be processed in each validation epoch. Default is None.
        """
        if dataset_generated is not None:
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            model.fit(
                dataset_generated,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs
            )
        else:
            print("Give Dataset")


    def save_model(self,model, filename='mymodel.h5'):
        """
        Save the Keras model to a file.

        Parameters:
        - model: Keras model to be saved.
        - filename (str): Name of the file to save the model to.
        """
        model.save(filename)
        print(f"Saved model as {filename}")
            


        

        
