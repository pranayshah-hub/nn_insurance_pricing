# -*- coding: utf-8 -*-
"""part3_pricing_model_linear

"""

import numpy as np
import pickle
import pandas as pd
import sklearn
import keras
from keras import Sequential
from keras.layers import LeakyReLU
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns

class PricingModel():
    def __init__(self, calibrate_probabilities=False):
        self.calibrate = calibrate_probabilities
        self.batch_size = 150
        self.epochs = 10
        self.X_raw = pd.read_csv('part3_data.csv') 
        self.claims_raw = np.array(pd.read_csv('part3_data.csv')['claim_amount'])
        self.y_raw = None
        self.X_test = None
        self.y_test = None
        self.y_mean = None
        self.S_x = 5.0/6
        self.base_classifier = None
        self.model = None

    # Logistic Regression Model
    def logistic(self):
        model = LogisticRegression(random_state=42, solver='lbfgs')
        return model

    # Ridge Regression Model
    def ridge(self):
        model = linear_model.Ridge(alpha = 0.5)
        return model

    def _preprocessor(self, X_raw, training = None):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        X: ndarray
            A clean data set that is used for training and prediction.
        """
        # =============================================================

        # Select Features
        # NB: 'claim_amount' and 'made_claim' already removed in hidden dataset, so no need to drop these
        if training == True:
            part3_df = X_raw.drop(['id_policy', 'drv_sex2', 'vh_make', 'vh_model', 'claim_amount'], axis = 1) 
        else:
            part3_df = X_raw.drop(['id_policy', 'drv_sex2', 'vh_make', 'vh_model'], axis = 1) 

        # For the feature 'pol_insee_code' we want to extract the first 2 digits as these are one of 96 departments:
        part3_df['pol_insee_code'] = part3_df['pol_insee_code'].astype(str).str[:2]
        # Replace NaN values with mthe mean of the feature
        part3_df[['town_mean_altitude', 'town_surface_area', 'population', 'commune_code', 'canton_code', 'city_district_code']] = part3_df[['town_mean_altitude', 'town_surface_area', 'population', 'commune_code', 'canton_code', 'city_district_code']].replace(np.nan, part3_df[['town_mean_altitude', 'town_surface_area', 'population', 'commune_code', 'canton_code', 'city_district_code']].mean())
        # Deal with NaN values in column of type 'object'
        part3_df.dropna(subset=['regional_department_code'], inplace=True)

        # Convert categorical variables into one-hot encoding
        part3_df_new = pd.get_dummies(part3_df, columns=['pol_coverage'])
        part3_df_new = pd.get_dummies(part3_df_new, columns=['pol_pay_freq', 'pol_payd'])
        part3_df_new = pd.get_dummies(part3_df_new, columns=['pol_usage', 'pol_insee_code'])
        part3_df_new = pd.get_dummies(part3_df_new, columns=['drv_drv2', 'drv_sex1'])
        part3_df_new = pd.get_dummies(part3_df_new, columns=['vh_fuel', 'vh_type'])
        part3_df_new = pd.get_dummies(part3_df_new, columns=['regional_department_code']) 

        # If we are using the preprocessor for training, then the training data is used:
        if training == True:
            # Move target column to the end
            columns = list(part3_df_new.columns.values)
            columns.pop(columns.index('made_claim'))
            part3_df_new = part3_df_new[columns+['made_claim']]
            y = pd.DataFrame(part3_df_new.iloc[:,-1:])
            self.y_raw = y
            X = pd.DataFrame(part3_df_new.iloc[:,:-1])
        else:
            X = pd.DataFrame(part3_df_new)

        # Normalise dataset
        X_unnormalised = X.values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()  
        X_unnormalised = min_max_scaler.fit_transform(X_unnormalised)      
        X_normalised = pd.DataFrame(min_max_scaler.fit_transform(X_unnormalised), columns=X.columns, index=X.index)

        # Set clean dataset
        if training == True:
            X_clean = pd.concat([X_normalised, self.y_raw], axis=1)
        else:
            X_clean = X_normalised

        return X_clean #X_train, X_test #NB X_train and X_test contain the target columns concatenated on the end
    
    def probability_model(self):
        model = Sequential()

        model.add(Dense(80, input_shape=(236,)))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.4))
        model.add(Dense(50))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.4))
        model.add(Dense(20))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def fit(self, X_raw, y_raw, claims_raw): 
        """Classifier training function.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded
        y_raw : ndarray
            A one dimensional array, this is the binary target variable
        claims_raw: ndarray
            A one dimensional array which records the severity of claims

        Returns
        -------
        self: (optional)
            an instance of the fitted model

        """
        nnz = np.where(claims_raw != 0)[0]
        self.y_mean = np.mean(claims_raw[nnz])
        # =============================================================
        # Preprocess the data first
        X_clean = self._preprocessor(X_raw, training=True)

        # Obtain Features and Target dataframes:  
        X = pd.DataFrame(X_clean.iloc[:,:-1])
        y = pd.DataFrame(X_clean.iloc[:,-1:])

        #Split dataset first 
        X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.05, shuffle = True)

        # Balance Dataset
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        X_cv, y_cv = sm.fit_resample(X_cv, y_cv)

        y_train = pd.DataFrame(y_train, columns=['made_claim'])
        X_train = pd.DataFrame(X_train, columns=X.columns)
        y_cv = pd.DataFrame(y_cv, columns=['made_claim'])
        X_cv = pd.DataFrame(X_cv, columns=X.columns)

        # Model Fitting and Evaluation
        threshold = 0.5
        # Add more linear models as necessary. Logistic Regression is tested here
        linear_models = ['logistic']
        for linear_model in linear_models:
            if linear_model == 'logistic':
                self.base_classifier = self.logistic()
                self.model = self.base_classifier
                self.model.fit(X_train, y_train)
                y_pred = np.round(self.model.predict(X_cv))
                cm = confusion_matrix(y_cv, y_pred)
                accuracy = ((y_pred == y_cv.to_numpy().T).sum())*1.0/len(y_cv)
                print('Confusion Matrix for Logistic Regression: \n')
                print(cm)
                print('Accuracy for Logistic Regression: \n')
                print(accuracy)
                self.base_classifier.fit(X_train, y_train)

        # Return Classifier
        return self.base_classifier

        if self.calibrate:
            self.base_classifier = fit_and_calibrate_classifier(
                self.base_classifier, X_clean, y_raw)
        else:
            self.base_classifier = self.base_classifier.fit(X_clean, y_raw)
        return self.base_classifier

    def predict_claim_probability(self, X_raw): 
        """Classifier probability prediction function.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # Preprocess data
        X_clean = self._preprocessor(X_raw, training=False)

        # Get trained model
        trained_model = self.fit(self.X_raw, None, self.claims_raw)

        #Claim Probability
        y_pred = trained_model.predict(X_clean)

        print(y_pred)

        return y_pred # return probabilities for the positive class (label 1)

    def predict_premium(self, X_raw):
        """Predicts premiums based on the pricing model.

        Parameters
        ----------
        X_raw : numpy.ndarray
            A numpy array, this is the raw data as downloaded

        Returns
        -------
        numpy.ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # Pricing Strategy: 
        self.y_mean = self.y_mean * self.S_x

        return self.predict_claim_probability(X_raw) * self.y_mean

    def save_model(self):
        """Saves the class instance as a pickle file."""
        # =============================================================
        with open('part3_pricing_model_linear.pickle', 'wb') as target:
            pickle.dump(self, target)