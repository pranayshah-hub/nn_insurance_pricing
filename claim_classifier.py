import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve


class ClaimClassifier:
    def __init__(self):
        self.batch_size = 256
        self.epochs = 800
        self.model = None
        self.mu = 0
        self.sigma = 1

    def find_distribution(self, X_raw):
        self.mu = np.mean(X_raw,axis = 0)
        self.sigma = np.std(X_raw, axis = 0)

    def _preprocessor(self, X_raw):
        X_clean = (X_raw - self.mu)/self.sigma
        return X_clean

    def baseline_model(self, layers = [128, 256, 512, 256, 128], lr = 0.001, dropout = 0.4):
        model = Sequential()
        model.add(Dense(layers[0], activation='relu', input_shape=(9,)))
        model.add(Dropout(dropout))
        for i in range(1,len(layers)-1):
            model.add(Dense(layers[i], activation='relu'))
            model.add(Dropout(dropout))
        model.add(Dense(layers[-1], activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        adamOpti = Adam(lr = lr)
        model.compile(optimizer=adamOpti,
              loss='binary_crossentropy',
              metrics=['accuracy'])
        return model

    def SMOTE(self,X,Y):
        K = 5
        index_0 = np.where(Y == 0)[0]
        index_1 = np.where(Y == 1)[0]
        X_new = np.take(X,index_0,axis = 0)
        Y_new = np.take(Y,index_0,axis = 0)
        increase_num = len(index_0)-len(index_1)
        count = 0
        while count <= increase_num:
            for i in index_1:
                current_x = X[i]
                distance = np.linalg.norm(X-current_x,axis = 1)
                idx = np.argpartition(distance, K)
                k_nearest_index = idx[:K]
                sample = np.random.choice(k_nearest_index,size = 1)
                X_interpolated = X[sample]+np.random.uniform()*(current_x-X[sample])
                X_new = np.vstack([X_new, X_interpolated])
                count += 1
        Y_new = np.concatenate([Y_new,np.ones(count)])
        print(np.mean(X_new,axis = 1))
        return X_new,Y_new



    def fit(self, X_train, y_train, X_test, y_test):
        # sm = imblearn.over_sampling.SMOTE()
        # X_train, y_train = sm.fit_resample(X_train, y_train)
        # print(X_train.shape)
        # print(y_train.shape)
        X_train, y_train = self.SMOTE(X_train, y_train)
        # X_train, y_train = self.balance_data(X_train, y_train)
        self.model.fit(X_train,y_train, validation_data = (X_test, y_test), batch_size = self.batch_size , epochs = self.epochs)
        

    def balance_data(self,X,Y):
        # #downsampling
        # index_0 = np.where(Y == 0)[0]
        # index_1 = np.where(Y == 1)[0]
        # index_0_downsampled = np.random.choice(index_0, size = len(index_1), replace=False)
        # indices = np.hstack((index_0_downsampled ,index_1))
        # np.random.shuffle(indices)
        # # for i in range(10):
        # #     print(X[indices[i]])
        # #     print(Y[indices[i]])
        # #     print(self.mu)
        # # print("____________________________")
        # X = np.take(X,indices,axis = 0)
        # Y = np.take(Y,indices,axis = 0)
        # # for i in range(10):
        # #     print(X[i])
        # #     print(Y[i])

        # upsampling
        index_0 = np.where(Y == 0)[0]
        index_1 = np.where(Y == 1)[0]
        index_1_upsampled = np.random.choice(index_1, size=int(len(index_0)), replace=True)
        indices = np.hstack((index_1_upsampled,index_0))
        np.random.shuffle(indices)
        X = np.take(X,indices,axis = 0)
        Y = np.take(Y,indices,axis = 0)
        return X,Y

    def predict(self, X_raw, threshold):
        # X_clean = self._preprocessor(X_raw)
        y_prob = self.model.predict(X_raw)
        return np.round(y_prob)
        count = 0
        y_pred =(y_pred>threshold)
        return y_pred

    def evaluate_architecture(self, X_raw, y_raw):
        self.find_distribution(X_raw)
        X_clean = self._preprocessor(X_raw)
        self.model = self.baseline_model()
        self.model.summary()
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_raw, test_size=0.1, shuffle = True)
        self.fit(X_train,y_train,X_test, y_test)
        threshold = 0.5
        y_pred = self.predict(X_test,threshold)
        y_pred = np.squeeze(y_pred, 1)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

    def fine_tuning(self, X, Y):
        param_grid = dict(layers=[[128, 256, 128, 64], ], lr=[0.001, 0.005, 0.01], dropout=[0.2, 0.3, 0.4])
        model = scikit_learn.KerasClassifier(build_fn=self.baseline_model, verbose=0)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, scoring='roc_auc')
        grid_result = grid.fit(X, Y)
        print('Best: {} using {}'.format(grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, std, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, std, param))
        self.model = self.baseline_model(grid_result.best_params_['layers'], grid_result.best_params_['lr'], grid_result.best_params_['dropout'])

    def save_model(self):
        with open("part2_claim_classifier.pickle", "wb") as target:
            pickle.dump(self, target)


def ClaimClassifierHyperParameterSearch():
    """Performs a hyper-parameter for fine-tuning the classifier.

    """

    pass

df = pd.read_csv('./part2_data.csv')

classifier = ClaimClassifier()
X= df.iloc[:,0:9].to_numpy()
Y= df.iloc[:,10].to_numpy()

classifier.evaluate_architecture(X,Y)




























