#!/usr/bin/env python
# coding: utf-8

# In[3]:


'''
Import the packages needed for classification
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
plt.close()
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as metrics
#import tensorflow as tf
#from tqdm import trange


# In[43]:


'''
Set directory parameters
'''
# Set the directories for the data and the CSV files that contain ids/labels
dir_train_images  = '/Users/Calvin/Desktop/2020Spring/IDS705_Machine_Learning/Kaggle Competition/ids705sp2020/training/'
dir_test_images   = '/Users/Calvin/Desktop/2020Spring/IDS705_Machine_Learning/Kaggle Competition/ids705sp2020/testing/'
dir_train_labels  = '/Users/Calvin/Desktop/2020Spring/IDS705_Machine_Learning/Kaggle Competition/ids705sp2020/labels_training.csv'
dir_test_ids      = '/Users/Calvin/Desktop/2020Spring/IDS705_Machine_Learning/Kaggle Competition/ids705sp2020/submission/sub3.csv'


# In[67]:


def load_data(dir_data, dir_labels, training=True):
    ''' Load each of the image files into memory 

    While this is feasible with a smaller dataset, for larger datasets,
    not all the images would be able to be loaded into memory

    When training=True, the labels are also loaded
    '''
    labels_pd = pd.read_csv(dir_labels)
    
    if training==False:
        scores = labels_pd.score.values
    ids       = labels_pd.id.values
    data      = []
    for identifier in ids:
        fname     = dir_data + identifier.astype(str) + '.tif'
        image     = mpl.image.imread(fname)
        data.append(image)
    data = np.array(data) # Convert to Numpy array
    if training:
        labels = labels_pd.label.values
        return data, labels
    else:
        return data, ids, scores

#Create index of 33*33 map    
iter_lst=[]
for j in range(0,33):
    for i in range(0,33):
        iter_lst.append((303*j)+3*i+102)

        
def new_feat_grey(vectorized_data):
    new_feat_mean=[]
    new_feat_max=[]
    for j in range(vectorized_data.shape[0]):
        for i in iter_lst:
            mean=np.mean([vectorized_data[j,i],vectorized_data[j,i+1],vectorized_data[j,i+2],vectorized_data[j,i+101],vectorized_data[j,i+102],vectorized_data[j,i+103],vectorized_data[j,i+202],vectorized_data[j,i+203],vectorized_data[j,i+204]])
            max=np.max([vectorized_data[j,i],vectorized_data[j,i+1],vectorized_data[j,i+2],vectorized_data[j,i+101],vectorized_data[j,i+102],vectorized_data[j,i+103],vectorized_data[j,i+202],vectorized_data[j,i+203],vectorized_data[j,i+204]])
            new_feat_mean.append(mean)
            new_feat_max.append(max)
            
    new_feat_mean=np.asarray(new_feat_mean).reshape(len(vectorized_data),-1)
    new_feat_max=np.asarray(new_feat_max).reshape(len(vectorized_data),-1)
            
    return new_feat_mean, new_feat_max
    
def new_feat_rgb(vectorized_data):
    new_feat_mean=[]
    new_feat_max=[]
    for j in range(vectorized_data.shape[0]):
        for i in iter_lst:
            mean_r=np.mean([vectorized_data[j,i,0],vectorized_data[j,i+1,0],vectorized_data[j,i+2,0],
                            vectorized_data[j,i+101,0],vectorized_data[j,i+102,0],vectorized_data[j,i+103,0],
                            vectorized_data[j,i+202,0],vectorized_data[j,i+203,0],vectorized_data[j,i+204,0]])
            mean_g=np.mean([vectorized_data[j,i,1],vectorized_data[j,i+1,1],vectorized_data[j,i+2,1],
                            vectorized_data[j,i+101,1],vectorized_data[j,i+102,1],vectorized_data[j,i+103,1],
                            vectorized_data[j,i+202,1],vectorized_data[j,i+203,1],vectorized_data[j,i+204,1]])
            mean_b=np.mean([vectorized_data[j,i,2],vectorized_data[j,i+1,2],vectorized_data[j,i+2,2],
                            vectorized_data[j,i+101,2],vectorized_data[j,i+102,2],vectorized_data[j,i+103,2],
                            vectorized_data[j,i+202,2],vectorized_data[j,i+203,2],vectorized_data[j,i+204,2]])
            
            max_r=np.max([vectorized_data[j,i,0],vectorized_data[j,i+1,0],vectorized_data[j,i+2,0],
                            vectorized_data[j,i+101,0],vectorized_data[j,i+102,0],vectorized_data[j,i+103,0],
                            vectorized_data[j,i+202,0],vectorized_data[j,i+203,0],vectorized_data[j,i+204,0]])
            max_g=np.max([vectorized_data[j,i,1],vectorized_data[j,i+1,1],vectorized_data[j,i+2,1],
                            vectorized_data[j,i+101,1],vectorized_data[j,i+102,1],vectorized_data[j,i+103,1],
                            vectorized_data[j,i+202,1],vectorized_data[j,i+203,1],vectorized_data[j,i+204,1]])
            max_b=np.max([vectorized_data[j,i,2],vectorized_data[j,i+1,2],vectorized_data[j,i+2,2],
                            vectorized_data[j,i+101,2],vectorized_data[j,i+102,2],vectorized_data[j,i+103,2],
                            vectorized_data[j,i+202,2],vectorized_data[j,i+203,2],vectorized_data[j,i+204,2]])
            
            new_feat_mean.append([mean_r,mean_g,mean_b])
            new_feat_max.append([max_r,max_g,max_b])
            
    new_feat_mean=np.asarray(new_feat_mean).reshape(len(vectorized_data),-1)
    new_feat_max=np.asarray(new_feat_max).reshape(len(vectorized_data),-1)
            
    return new_feat_mean, new_feat_max


def preprocess_and_extract_features_grey(data):
    #Preprocess data and extract features
    
    # Make the image grayscale
    data = np.mean(data, axis=3)
    
    # Vectorize the grayscale matrices
    vectorized_data = data.reshape(data.shape[0],-1)
    
    # Extract features (33*33 grey)
    train_feat_mean, train_feat_max = new_feat_grey(vectorized_data)
    
    return train_feat_mean, train_feat_max

def preprocess_and_extract_features_rgb(data):
    '''Preprocess data and extract features
    
    Preprocess: normalize, scale, repair
    Extract features: transformations and dimensionality reduction
    '''
    
    # Vectorize 
    vectorized_data = data.reshape(len(data),10201,3)
    
    # Extract features (33*33 RGB) 
    train_feat_mean, train_feat_max = new_feat_rgb(vectorized_data)
    
    return train_feat_mean, train_feat_max

def set_classifier_Knn():
    '''Shared function to select the classifier for both performance evaluation
    and testing
    '''
    return KNeighborsClassifier(n_neighbors=3)

def set_classifier_logistic_l1():
    '''Shared function to select the classifier for both performance evaluation
    and testing
    '''
    log_reg=LogisticRegression(penalty='l1',C=10,solver="liblinear")
    
    return log_reg

def set_classifier_logistic_l2():
    '''Shared function to select the classifier for both performance evaluation
    and testing
    '''
    log_reg=LogisticRegression()
    
    return log_reg

def set_classifier_GradientBoostingClassifier():
    
    '''Shared function to select the classifier for both performance evaluation
    and testing
    '''
    gb=GradientBoostingClassifier(loss='exponential')
    
    return gb

def set_classifier_BaggingClassifier():
    
    '''Shared function to select the classifier for both performance evaluation
    and testing
    '''
    bagging=BaggingClassifier()
    
    return bagging

def set_classifier_RandomForestClassifier():
    
    '''Shared function to select the classifier for both performance evaluation
    and testing
    '''
    rf=RandomForestClassifier()
    
    return rf

def cv_performance_assessment(X,y,k,clf):
    '''Cross validated performance assessment
    
    X   = training data
    y   = training labels
    k   = number of folds for cross validation
    clf = classifier to use
    
    Divide the training data into k folds of training and validation data. 
    For each fold the classifier will be trained on the training data and
    tested on the validation data. The classifier prediction scores are 
    aggregated and output
    '''
    # Establish the k folds
    prediction_scores = np.empty(y.shape[0],dtype='object')
    kf = StratifiedKFold(n_splits=k, shuffle=True)
    for train_index, val_index in kf.split(X, y):
        # Extract the training and validation data for this fold
        X_train, X_val   = X[train_index], X[val_index]
        y_train          = y[train_index]
        
        # Train the classifier
        #X_train_features = preprocess_and_extract_features(X_train)
        clf              = clf.fit(X_train,y_train)
        
        # Test the classifier on the validation data for this fold
        #X_val_features   = preprocess_and_extract_features(X_val)
        cpred            = clf.predict_proba(X_val)
        #print (cpred[])
        # Save the predictions for this fold
        prediction_scores[val_index] = cpred[:,1]
    return prediction_scores

def predict_test(X,y,test_x,clf):
    
    clf              = clf.fit(X,y)
        
    cpred            = clf.predict_proba(test_x)

    return cpred[:,1]

def plot_roc(labels, prediction_scores):
    fpr, tpr, _ = metrics.roc_curve(labels, prediction_scores, pos_label=1)
    auc = metrics.roc_auc_score(labels, prediction_scores)
    legend_string = 'AUC = {:0.3f}'.format(auc)
   
    plt.plot([0,1],[0,1],'--', color='gray', label='Chance')
    plt.plot(fpr, tpr, label=legend_string)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid('on')
    plt.axis('square')
    plt.legend()
    plt.tight_layout()


# In[22]:


train_data, train_labels = load_data(dir_train_images,dir_train_labels, True)


# In[23]:


train_feat_mean, train_feat_max=preprocess_and_extract_features_rgb(train_data)


# In[44]:


test_data, id, test_scores =  load_data(dir_test_images,dir_test_ids, False)


# In[51]:


test_feat_mean, test_feat_max=preprocess_and_extract_features_rgb(test_data)


# In[ ]:


test_labels=[]
for i in test_scores:
    if i >= 0.5:
        test_labels.append(1)
    else:
        test_labels.append(0)
test_labels=np.asarray(test_labels)


# In[31]:


pred_score=cv_performance_assessment(train_feat_mean,train_labels,5,set_classifier_GradientBoostingClassifier())
#AUC=0.762


# In[29]:


pred_score=cv_performance_assessment(train_feat_max,train_labels,5,set_classifier_GradientBoostingClassifier())
#AUC=0.749


# In[34]:


pred_score=cv_performance_assessment(train_feat_mean,train_labels,5,set_classifier_logistic_l1())
#0.67


# In[ ]:


pred_score=cv_performance_assessment(train_feat_mean,train_labels,5,set_classifier_logistic_l2())
#0.685


# In[39]:


pred_score=cv_performance_assessment(train_feat_mean,train_labels,5,set_classifier_RandomForestClassifier())
#0.754


# In[41]:


pred_score=cv_performance_assessment(train_feat_max,train_labels,5,set_classifier_RandomForestClassifier())
#0.75


# In[68]:


test_pred = predict_test(train_feat_mean,train_labels,test_feat_mean,set_classifier_GradientBoostingClassifier())
#0.785


# In[61]:


test_pred = predict_test(train_feat_max,train_labels,test_feat_max,set_classifier_GradientBoostingClassifier())
#0.760


# In[63]:


test_pred = predict_test(train_feat_mean,train_labels,test_feat_mean,set_classifier_RandomForestClassifier())
#0.777


# In[65]:


test_pred = predict_test(train_feat_max,train_labels,test_feat_max,set_classifier_RandomForestClassifier())
#0.769


# In[69]:


plot_roc(test_labels,test_pred)

