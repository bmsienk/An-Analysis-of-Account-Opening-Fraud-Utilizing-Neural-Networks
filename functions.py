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
        y_pred = tf.greater_equal(y_pred, self.threshold)  # Using default threshold of 0.5 to call a prediction as positive labeled.
        
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
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(25, 15))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(train_loss, color='tab:blue', label='Train Loss')
    ax1.plot(valid_loss, color='tab:orange', label='Valid Loss')
    ax1.legend(loc='upper left')
    plt.title('Model Loss')

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.plot(train_auc, color='tab:blue', label='Train AUC')
    ax2.plot(valid_auc, color='tab:orange', label='Valid AUC')
    ax2.legend(loc='upper left')
    ax2.set_ylim([0.5,1.05])
    plt.title('Model AUC')
        
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

    cm = confusion_matrix(y, y_pred_adjusted)
    cm_df = pd.DataFrame(cm)

    if sum(cm_df[1]) == 0:
    
        percentages = [(cm_df[0][0]/sum(cm_df[0])).round(4),
                       (cm_df[1][0]/1),
                       (cm_df[0][1]/sum(cm_df[0])).round(4),
                       (cm_df[1][1]/1)
                      ]

    else:

        percentages = [(cm_df[0][0]/sum(cm_df[0])).round(4),
                       (cm_df[1][0]/sum(cm_df[1])).round(4),
                       (cm_df[0][1]/sum(cm_df[0])).round(4),
                       (cm_df[1][1]/sum(cm_df[1])).round(4)
                      ]

    sns.heatmap(cm, ax=ax3, annot=True, cmap='Blues', fmt='0.7g') 
    i=0
    for t in ax3.texts:
        t.set_text(t.get_text() + f" ({percentages[i]*100}%)")
        i+=1

    plt.sca(ax3)
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')
    
    train_recall=[value for key, value in history.items() if 'recall' in key.lower()][0]
    valid_recall=[value for key, value in history.items() if 'recall' in key.lower()][1]
    train_precision=[value for key, value in history.items() if 'precision' in key.lower()][0]
    valid_precision=[value for key, value in history.items() if 'precision' in key.lower()][1]
    train_fpr=[value for key, value in history.items() if 'false_positive_rate' in key.lower()][0]
    valid_fpr=[value for key, value in history.items() if 'false_positive_rate' in key.lower()][1]
    
    if (cm_df[0][1] == 0) or (cm_df[1][1] ==0):
        train_f1 = 'N/A'
        valid_f1 = 'N/A'
    else:
        train_f1 = 2*(train_recall[-1]*train_precision[-1])/(train_recall[-1]+train_precision[-1])
        valid_f1 = 2*(valid_recall[-1]*valid_precision[-1])/(valid_recall[-1]+valid_precision[-1])
    
    print(f"\n Train f1: {train_f1} \n Val f1: {valid_f1} \n\n Train Recall: {train_recall[-1]} \n Val Recall: {valid_recall[-1]} \n\n Train FPR: {train_fpr[-1]} \n Val FPR: {valid_fpr[-1]}")

    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Recall')
    ax4.set_ylim([-0.05,1.05])
    ax4.plot(train_recall, '--', color='tab:orange', label='Train Recall')
    ax4.plot(valid_recall, color='tab:orange', label='Valid Recall')
    ax4.legend(loc='lower left')
    plt.title('Model Recall and FPR')
    ax5 = ax4.twinx()  # instantiate a second axes that shares the same x-axis
    ax5.set_ylabel('False Positive Rate')  # we already handled the x-label with ax1
    ax5.plot(train_fpr, '--', color='tab:blue', label='Train FPR')
    ax5.plot(valid_fpr, color='tab:blue', label='Valid FPR')
    ax5.set_ylim([-0.05,1.05])
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax5.legend(loc='lower right')
    plt.show()
    
def create_class_weight(labels_dict,mu=0.15):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()
    
    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
    
    return class_weight

def apply_model_thresholds(model_function, thresholds=[0.5, 0.4, 0.3, 0.2, 0.1]):

    for t in thresholds:
        model_function()