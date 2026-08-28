#So that discretisation and setup does not need to be repeated each time DeepHit is instantiated
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from pycox.models import DeepHitSingle
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
import torchtuples as tt

class DeepHit:
    def __init__(self, X_train, X_test, y_train, y_test, labtrans=None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        #extract durations and events from structured array
        self.durations_train = self.y_train['time']
        self.events_train = self.y_train['event'].astype(np.float32) #convert from boolean to float since DeepHit reads numerical inputs
        self.durations_test = self.y_test['time']
        self.events_test = self.y_test['event'].astype(np.float32)

        #standardize non categorical features (so each feature has mean 0 and variance 1)
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        numerical_cols = X_train.columns.difference(categorical_cols)

        self.scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        X_test_scaled[numerical_cols] = self.scaler.transform(X_test[numerical_cols])

        #convert to float32 for PyTorch
        self.X_train_scaled = X_train_scaled.values.astype('float32')
        self.X_test_scaled = X_test_scaled.values.astype('float32')
        #.values part converts data frame to numpy array, so it avoids errors when fitting model

        #discretize time into bins (DeepHit requirement)
        if labtrans is not None:
            #Use shared pre-fitted discretizer (e.g. from bagging)
            self.labtrans = labtrans
            self.y_train_discrete = self.labtrans.transform(self.durations_train, self.events_train)
        else:
            #Create and fit a new discretizer
            num_durations = 100 #number of bins
            self.labtrans = LabTransDiscreteTime(num_durations) #converts survival_months to bin index (which bin it falls into)
            self.y_train_discrete = self.labtrans.fit_transform(self.durations_train, self.events_train)

        #y_train_discrete is a list of indexes that each survival month belongs to, along with event status
        self.index_of_train_duration, self.events = self.y_train_discrete

        #Create neural network
        number_of_features = self.X_train_scaled.shape[1]

        #MLP stands for Multi-Layer Perceptron (standard feedforward neural network)
        #creates neural network with fully connected layers
        net = tt.practical.MLPVanilla(
            in_features=number_of_features,
            #hidden layer sizes
            num_nodes=[256, 128, 64], #output is 101 neurons - one per bin
            out_features=self.labtrans.out_features, #101 - number of output neurons
            batch_norm=True, #normalise activation between layers - output values of neurons
            dropout=0.3 #randomly turn of 30% neurons in each layer to reduce overfitting
        )
        self.model = DeepHitSingle(
            net,
            #Optimiser that trains neural network by adjusting its weights
            #Uses momentum and RMSprop
            tt.optim.Adam(lr=0.001), #lr controls how big the weight updates are

            #hyper parameter that controls the value of likelihood loss (survival bin)
            alpha=0.2,
            #controls value of ranking loss (discrimination based on feature)
            sigma=0.1,
            duration_index=self.labtrans.cuts #bin edges
        )

    def fit(self):

        log = self.model.fit(
        self.X_train_scaled,
        self.y_train_discrete,
        batch_size=1024,
        epochs=10,
        verbose=True
        )

    def predict_surv(self):
        surv = self.model.predict_surv_df(self.X_test_scaled)
        return surv,self.model.duration_index