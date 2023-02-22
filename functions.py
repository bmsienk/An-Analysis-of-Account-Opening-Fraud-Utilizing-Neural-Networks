import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras import datasets, layers, models, regularizers
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier
import math

#relevant imports

class FalsePositiveRate(tf.keras.metrics.Metric):
    def __init__(self, name='false_positive_rate', threshold=0.5, **kwargs):
        super(FalsePositiveRate, self).__init__(name=name, **kwargs)
        self.negatives = self.add_weight(name='negatives', initializer='zeros')
        self.false_positives = self.add_weight(name='false_negatives', initializer='zeros')
        self.threshold = threshold
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        '''
        Arguments:
        y_true  The actual y. Passed by default to Metric classes.
        y_pred  The predicted y. Passed by default to Metric classes.
        
        '''
        # Compute the number of negatives.
        
        y_true = tf.cast(y_true, tf.bool)
        
        negatives = tf.reduce_sum(tf.cast(tf.equal(y_true, False), self.dtype))
        
        self.negatives.assign_add(negatives)
        
        # Compute the number of false positives.
        
        y_pred = tf.greater_equal(y_pred, self.threshold) 
        
        # Using default threshold of 0.5 to call a prediction as positive labeled.
        
        false_positive_values = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True)) 
        false_positive_values = tf.cast(false_positive_values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_weights(sample_weight, values)
            values = tf.multiply(false_positive_values, sample_weight)
        
        false_positives = tf.reduce_sum(false_positive_values)
        
        self.false_positives.assign_add(false_positives)
        
    def result(self):
        return tf.divide(self.false_positives, self.negatives)
    
def evaluate(model, name, history, X, y, threshold=0.5):
    
    print(f"Results for {name} with threshold = {threshold}.")
    
    plt.rcParams.update({'font.size': 18})
    
    #Create a function that provides useful vis for model
    #performance. This is especially useful as we are most
    #concerned with the number of false negatives
    
    train_loss=[value for key, value in history.items() if 'loss' in key.lower()][0]
    valid_loss=[value for key, value in history.items() if 'loss' in key.lower()][1]
    train_auc=[value for key, value in history.items() if 'auc' in key.lower()][0]
    valid_auc=[value for key, value in history.items() if 'auc' in key.lower()][1]
    
    #creating lists of loss and auc for visualizations
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(25, 15))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(train_loss, color='tab:blue', label='Train Loss')
    ax1.plot(valid_loss, color='tab:orange', label='Valid Loss')
    ax1.legend(loc='upper left')
    plt.title('Model Loss')
    
    #plotting train and validation loss

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.plot(train_auc, color='tab:blue', label='Train AUC')
    ax2.plot(valid_auc, color='tab:orange', label='Valid AUC')
    ax2.legend(loc='upper left')
    ax2.set_ylim([0.5,1.05])
    plt.title('Model AUC')
    
    #plotting train and validation auc
        
    y_pred = model.predict(X)
    y_pred_adjusted = np.zeros([len(y), ])
    i=0
    for pred in y_pred:
        if pred > threshold:
            y_pred_adjusted[i] = 1
            i+=1
        else:
            y_pred_adjusted[i] = 0
            i+=1
            
    #creating a for loop for adjusted predictions based on the given threshold

    cm = confusion_matrix(y, y_pred_adjusted)
    cm_df = pd.DataFrame(cm)
    
    #creating a confusion matrix based on the adjusted predictions

    sns.heatmap(cm, ax=ax3, annot=True, cmap='Blues', fmt='0.7g') 
    plt.sca(ax3)
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')
    
    #plotting the confusion matrix
    
    train_recall=[value for key, value in history.items() if 'recall' in key.lower()][0]
    valid_recall=[value for key, value in history.items() if 'recall' in key.lower()][1]
    train_precision=[value for key, value in history.items() if 'precision' in key.lower()][0]
    valid_precision=[value for key, value in history.items() if 'precision' in key.lower()][1]
    train_fpr=[value for key, value in history.items() if 'false_positive_rate' in key.lower()][0]
    valid_fpr=[value for key, value in history.items() if 'false_positive_rate' in key.lower()][1]
    
    #creating lists for recall, precision, and fpr
    
    if (cm_df[0][1] == 0) or (cm_df[1][1] ==0):
        train_f1 = 'N/A'
        valid_f1 = 'N/A'
    else:
        train_f1 = 2*(train_recall[-1]*train_precision[-1])/(train_recall[-1]+train_precision[-1])
        valid_f1 = 2*(valid_recall[-1]*valid_precision[-1])/(valid_recall[-1]+valid_precision[-1])
        
    #a simple loop to avoid dividing by 0 when calculating the f1 score
    
    print(f"\n Train f1: {train_f1} \n Val f1: {valid_f1} \n\n Train Recall: {train_recall[-1]} \n Val Recall: {valid_recall[-1]} \n\n Train FPR: {train_fpr[-1]} \n Val FPR: {valid_fpr[-1]}")
    
    #printing out valuable metrics

    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Recall')
    ax4.set_ylim([-0.05,1.05])
    ax4.plot(train_recall, '--', color='tab:orange', label='Train Recall')
    ax4.plot(valid_recall, color='tab:orange', label='Valid Recall')
    ax4.legend(loc='lower left')
    plt.title('Model Recall and FPR')
    
    #plotting the recall
    
    ax5 = ax4.twinx()  

    #instantiate a second axes that shares the same x-axis
    
    ax5.set_ylabel('False Positive Rate') 
    ax5.plot(train_fpr, '--', color='tab:blue', label='Train FPR')
    ax5.plot(valid_fpr, color='tab:blue', label='Valid FPR')
    ax5.set_ylim([-0.05,1.05])
    fig.tight_layout()
    ax5.legend(loc='lower right')
    plt.show()
    
    #addition of fpr to recall plot
    
def create_class_weight(labels_dict,mu=0.15):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()
    
    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
        
    #simple function to calculate class weights based on a mu value
    
    return class_weight
        
class final_model():
    
    #creation of a final model class

    def __init__(self,
                 X_train,
                 y_train,
                 X_test,
                 y_test,
                 n_features,
                 class_weight,
                 name, 
                 epochs=50,
                 batch_size=256,
                 threshold=0.5):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_features = n_features
        self.class_weight = class_weight
        self.name = name
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = threshold
    
        self.test_recall = None
        self.test_fpr = None
        self.fpr_list = None
        self.recall_list = None
        self.cm_list = None
        
        #the class is instantiated with relevant modelling metrics
    
    def run_evaluate(self,
                     epochs=50,
                     batch_size=256,
                     threshold=0.5):
        
        #defining a class function to run and evaluate the final model
        
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)
        
        fpr_list = []
        recall_list = []
        cm_list = []
        
        print(f"Results for {self.name} with threshold = {threshold}.")

        for i in range(31):

            final_model = models.Sequential()

            final_model.add(layers.Input(shape=(self.n_features, )))
            final_model.add(layers.Dense(32,
                                         kernel_initializer='lecun_normal',
                                         activation='selu',
                                         kernel_regularizer=regularizers.L2()))
            final_model.add(layers.AlphaDropout(0.25))
            final_model.add(layers.Dense(16,
                                         kernel_initializer='lecun_normal',
                                         activation='selu',
                                         kernel_regularizer=regularizers.L2()))
            final_model.add(layers.AlphaDropout(0.25))
            final_model.add(layers.Dense(8,
                                         kernel_initializer='lecun_normal',
                                         activation='selu',
                                         kernel_regularizer=regularizers.L2()))
            final_model.add(layers.AlphaDropout(0.25))
            final_model.add(layers.Dense(4,
                                         kernel_initializer='lecun_normal',
                                         activation='selu',
                                         kernel_regularizer=regularizers.L2()))
            final_model.add(layers.Dense(1, activation='sigmoid'))


            final_model.compile(optimizer='adam',
                             loss='binary_crossentropy',
                             metrics=['AUC',
                                      tf.keras.metrics.Precision(thresholds=threshold),
                                      tf.keras.metrics.Recall(thresholds=threshold),
                                      FalsePositiveRate(threshold=threshold)])

            final_model_history = final_model.fit(self.X_train,
                                                  self.y_train,
                                                  batch_size=batch_size,
                                                  validation_data=(self.X_test, self.y_test),
                                                  class_weight=self.class_weight,
                                                  epochs=epochs, 
                                                  callbacks=[early_stop],
                                                  verbose=0).history
            
            #the above loop takes the model and runs it 30 times

            test_recall=[value for key, value in final_model_history.items() if 'recall' in key.lower()][1]
            test_fpr=[value for key, value in final_model_history.items() if 'false_positive_rate' in key.lower()][1]
            
            
            self.test_recall=test_recall
            self.test_fpr=test_fpr
            
            #the recall and fpr scores are made into a list and set as a metric
            #for the class

            fpr_list.append(test_fpr[-1])
            recall_list.append(test_recall[-1])
            
            #the final epoch value from the fpr and recall list is added to fpr_list
            #and recall_list

            y_pred = final_model.predict(self.X_test)
            y_pred_adjusted = np.zeros([len(self.y_test), ])
            j=0
            for pred in y_pred:
                if pred > threshold:
                    y_pred_adjusted[j] = 1
                    j+=1
                else:
                    y_pred_adjusted[j] = 0
                    j+=1
                    
            #same process as mentioned above for adjusted predictions

            cm = confusion_matrix(self.y_test, y_pred_adjusted)

            cm_list.append(np.array(cm))
        
        self.fpr_list = fpr_list
        self.recall_list = recall_list
        self.cm_list = cm_list
        
        #relevant metric lists set equal to the relevant class attributes
        
        print(f"\n\n Final Test Recall: {np.average(fpr_list)} \n\n Final Test FPR: {np.average(recall_list)}")
        
        #prints out relevant metrics

        fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize=(10, 16))

        ax1.scatter(fpr_list, recall_list)
        ax3 = ax1.twinx()
        ax3.scatter(np.average(fpr_list), np.average(recall_list), marker ="D", edgecolor ='black', s=80)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('Recall')
        plt.title('FPR vs. Recall')
        
        #plotting the fpr and recall scores for each iteration of the model
        #also plots the average of all 30 models
        
        sns.heatmap((sum(cm_list)/len(cm_list)).round(), ax=ax2, annot=True, cmap='Blues', fmt='0.7g') 
        plt.title('Averages Results')
        plt.show()
        
        #plotting the average confusion matrix for all 30 iterations