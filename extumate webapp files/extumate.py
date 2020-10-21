#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import joblib
import lime
import lime.lime_tabular


# In[2]:

st.cache()
preprocessor = joblib.load('reintubate_preprocessor_strip.sav')
clf = joblib.load("reintubate_model_strip.sav")

scaler_object = preprocessor.named_transformers_['num']
mean = scaler_object['scaler'].mean_
var = scaler_object['scaler'].var_

# In[3]:


#df_columns=['time_on_vent', 'anchor_age', 'heartrate', 'weight', 'hco3',
 #      'creatinine', 'bun', 'height', 'tidalvolume', 'temp', 'pulseox','re_intub_class', 'gender','tidal_weight']
df_columns=['time_on_vent', 'anchor_age', 'heartrate', 'weight', 'hco3', 'pulseox',
       'creatinine', 'bun', 'height', 'tidalvolume', 'temp', 're_intub_class',
       'gender', 'tidal_weight']

# In[4]:


st.set_option('deprecation.showfileUploaderEncoding', False)


# In[5]:


st.title('Extu-Mate: Helping ICU doctors decide when to extubate')

st.sidebar.subheader('Enter patient info:')
time_on_vent = st.sidebar.number_input(label = 'How long has the patient already been on the ventilator? (hours):',value=91)
time_on_vent = np.log(time_on_vent)

anchor_age = st.sidebar.number_input(label = 'Patient age (years):', value = 62)
anchor_age = np.log(anchor_age)

gender = st.sidebar.radio(label = 'Patient gender:', options  = ['M', 'F'])
weight = st.sidebar.number_input(label = 'Patient weight (lb):', value = 182)
height = st.sidebar.number_input(label = 'Patient height (inches):',value = 67)

st.sidebar.subheader("Enter patient's most recent vital signs:")
pulseox = st.sidebar.number_input(label = 'Oxygen saturation (%):', value = 99)
pulseox = np.log(pulseox)

heartrate = st.sidebar.number_input(label = 'Heart rate (bpm):', value = 86)
tidalvolume = st.sidebar.number_input(label = 'Tidal volume (mL):', value = 200)
temp = st.sidebar.number_input(label = 'Temperature (Celcius):', value = 37.06)

st.sidebar.subheader("Enter patient's most recent lab values:")
hco3 = st.sidebar.number_input(label = 'HCO3 (mEq/L):', value = 25.15)
creatinine = st.sidebar.number_input(label = 'Creatinine (mg/dL):', value = 1.24)
creatinine = np.log(creatinine+1)

bun = st.sidebar.number_input(label = 'Blood urea nitrogen (mg/dL):',value = 10)
bun = np.log(bun+1)

#tidal_weight = tidalvolume/weight
tidal_weight = tidalvolume/weight

re_intub_class = 0

st.cache()
test_data = np.array([[time_on_vent, anchor_age, heartrate, weight, hco3, pulseox, creatinine, bun,
       height, tidalvolume, temp,  
       re_intub_class, gender, tidal_weight]])

df = pd.DataFrame(data = test_data, columns=df_columns)
#x = df[df.columns.drop(['re_intub_class'])]
#x_columns = x.columns

df.drop('re_intub_class',axis=1,inplace=True)
df_scaled = preprocessor.transform(df)
sample_df = df_scaled.copy()
sample_test = df_scaled.flatten().reshape(1,-1)
#sample_test = sample_df.drop(labels=['re_intub_class'],axis=1).values

clf.predict(sample_df)
prediction_percent = np.int(clf.predict_proba(sample_test)[0][1]*100)
sentence = 'If you take your patient off the ventilator now, there is a '+ str(prediction_percent)+'% chance that they will need to be reintubated'

st.header(sentence)


X_train = pd.read_feather("strip_train_data")
X_scaled = preprocessor.transform(X_train)

categs= preprocessor.named_transformers_['cat']['onehot']
onehot_features = categs.get_feature_names()
numeric_features = preprocessor.transformers[0][2]
feature_names = np.concatenate((numeric_features.tolist(),onehot_features))
feature_names_polished = ['Ventilation time',  'Age of patient','Heart rate', 'Weight', 'HCO3 levels',
       'O2 saturation','Creatinine levels', 'Blood urea nitrogen levels', 'Height', 'Tidal Volume', 'Temperature',
       'Tidal volume normalized to weight', 'Gender']

zip_iterator = zip(feature_names, feature_names_polished)
feature_dict = dict(zip_iterator)

zip_mean= zip(feature_names, mean)
mean_dict = dict(zip_mean)

zip_var= zip(feature_names, var)
var_dict = dict(zip_var)

explainer = lime.lime_tabular.LimeTabularExplainer(X_scaled,  
                              feature_names=feature_names,  
                              #class_names=['re_intub_class'], 
                              #categorical_features=categorical_features ,
                              verbose=True, 
                              mode='classification',
                              discretize_continuous=True,
                              random_state = 101)

st.cache()
explog = explainer.explain_instance(sample_test[0,:], clf.predict_proba, num_samples = 100, num_features=5)
#explog.show_in_notebook(show_table=True)

feature_list = explog.as_list()
num_top_feats = len(feature_list)

printing_features = ''
if prediction_percent > 50:
    st.subheader("The likelihood that the patient will need to be reintubated can be explained by the following patient attributes:")
    j = 0
    for j in np.arange(num_top_feats):
            salient_feature = feature_list[j][0].split(' ')
            j = j+1
            for i in salient_feature:
                if i in feature_names:
                    explainable_feature = feature_dict[i]
                    printing_features = printing_features + explainable_feature + ', '
                    #st.write(explainable_feature)
else:
    st.write("The likelihood that the patient will need to be reintubated can be explained by the following patient attributes:")
    j = 0
    for j in np.arange(num_top_feats):
            salient_feature = feature_list[j][0].split(' ')
            j = j+1
            for i in salient_feature:
                if i in feature_names:
                    explainable_feature = feature_dict[i]
                    printing_features = printing_features + explainable_feature + ', '
                    #st.write(explainable_feature)
st.write(printing_features[:-2])                    
    
    
spiller_words = ['<','=','>','=>','>=','<=','=<']   
changeable_features = ['heartrate', 'hco3', 'creatinine', 'bun',
       'tidalvolume', 'temp', 'pulseox']

units_ref = ['(bpm)','(mEq/L)','(mg/dL)','(mg/dL)','(mL)','(Celcius)','(%)']

zip_units= zip(changeable_features, units_ref)
units_dict = dict(zip_units)

st.subheader("This prediction is driven by feature values in the ranges below.")

j = 0
for j in np.arange(num_top_feats):
    salient_feature = feature_list[j][0].split(' ')
    j = j+1
    
    for feature in salient_feature:
        if feature in changeable_features:
            x = ''
            for i in salient_feature:
                if i in spiller_words:
                    print(i)
                    value = i
                    x = x+value+ ' '
                else:
                    if i in changeable_features:
                        print(feature_dict[feature])
                        value = feature_dict[feature]
                        x = x+value + ' '
                    else:
                        if (i not in feature_names) & (i not in spiller_words):
                            if (feature=='bun')|(feature=='creatinine')|(feature=='pulseox'):
                                #st.write(feature_dict[feature])
                                #st.write(np.exp((float(i)*np.sqrt(var_dict[feature]))+ mean_dict[feature])-1)
                                #print(feature_dict[feature])
                                print(np.exp((float(i)*np.sqrt(var_dict[feature]))+ mean_dict[feature])-1)
                                value = "{:.2f}".format(np.exp((float(i)*np.sqrt(var_dict[feature]))+ mean_dict[feature])-1)
                                x = x+value + ' '
                            else:
                                #st.write(feature_dict[feature])
                                #st.write((float(i)*np.sqrt(var_dict[feature]))+ mean_dict[feature])
                                #print(feature_dict[feature])
                                print((float(i)*np.sqrt(var_dict[feature]))+ mean_dict[feature])
                                value = (float(i)*np.sqrt(var_dict[feature]))+ mean_dict[feature]
                                value = "{:.2f}".format((float(i)*np.sqrt(var_dict[feature]))+ mean_dict[feature])
                                x = x+value + ' '
            x_unit = units_dict[feature]
            x = x+x_unit+ ' '
            print(x)
            st.write(x) 
        
              
                            
        
# In[6]:


# =============================================================================
# csv_file = st.file_uploader(
#     label="Upload a csv file containing your patient's data.", type=["csv"], encoding="utf-8"
# )
# 
# if csv_file is not None:
#     df = pd.read_csv(csv_file)
#     x = df[mask]
#     x_scaled = scaler.transform(x)
#     sample_df = df.copy()
#     sample_df[mask] = x_scaled.flatten()
#     sample_test = sample_df.drop(labels=['re_intub_class'],axis=1).values
#     logmodel.predict(sample_test)
#     prediction_percent = logmodel.predict_proba(sample_test)[0,0]
#     st.write('There is a ', prediction_percent,
#     '% likelihood that extubation will be successful')
#     #st.dataframe(df)
# =============================================================================


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




