#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from tkinter import ttk
import os
from math import sqrt
import pandas as pd
import numpy as np
import seaborn as sns

import openai
from openai import OpenAI


# In[95]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[171]:


class UIgenerator:
    
    def __init__(self,size,title):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(size)

    def add_label(self,text, loc, padding):
        user_input_label = ttk.Label(self.root, text=text)
        user_input_label.grid(row=loc[0], column=loc[1], sticky=tk.W, padx=padding[0],pady=padding[1])

    def add_entry(self,input_width, loc, padding):
        user_input_entry = ttk.Entry(self.root, width=input_width)
        user_input_entry.grid(row=loc[0], column=loc[1], padx=padding[0],pady=padding[1])
        return user_input_entry

    def add_button(self,text,func,loc,colspan,padding):
        button = ttk.Button(self.root,text=text, command=func)
        button.grid(row=loc[0],column=loc[1],columnspan=colspan,padx=padding[0],pady=padding[1])
        return button

    '''def add_button_for_model_training(self,text,model_obj,data_path,loc,colspan,padding):
        button = ttk.Button(self.root,text=text, command=lambda: fetch_data_and_train(data_path, model_obj))
        button.grid(row=loc[0],column=loc[1],columnspan=colspan,padx=padding[0],pady=padding[1])
        return button'''
        
    def add_text(self,dim,state,loc,colspan,padding):
        text = tk.Text(self.root,height=dim[1],width=dim[0],state=state)
        text.grid(row=loc[0],column=loc[1],columnspan=colspan,padx=padding[0],pady=padding[1])
        return text

    def set_gpt_resp_text(self,text):
        self.gpt_text = text

    def set_gpt_input_entry(self,entry):
        self.gpt_input_entry = entry

    def set_pred_text(self,text):
        self.pred_text = text

    def set_model_entry_fields(self,entries):
        self.input_entry_avg_area_income = entries[0]
        self.input_entry_avg_area_house_age = entries[1]
        self.input_entry_avg_area_no_of_rooms = entries[2]
        self.input_entry_avg_area_no_of_bedrooms = entries[3]
        self.input_entry_area_population = entries[4]

    def set_model_obj(self,obj):
        self.model_obj = obj

    def set_data_path(self,dataset_path):
        self.dataset_path= dataset_path

    def set_openai_obj(self,obj):
        self.openai_obj = obj

    def fetch_data_and_train(self):
        if not self.model_obj.is_model_trained:
            df = pd.read_csv(self.dataset_path)
            X,y=self.model_obj.preprocess_data(df)
            X_train, X_test, y_train, y_test=self.model_obj.do_dataset_split(X,y)
            self.model_obj.train_model(X_train,y_train)
            #analyze on test To-do
        
    def compute_prediction(self):
        x_test = {"Avg. Area Income":[float(self.input_entry_avg_area_income.get())],
                  "Avg. Area House Age":[float(self.input_entry_avg_area_house_age.get())],
                  "Avg. Area Number of Rooms":[float(self.input_entry_avg_area_no_of_rooms.get())],
                  "Avg. Area Number of Bedrooms":[float(self.input_entry_avg_area_no_of_bedrooms.get())],
                  "Area Population":[float(self.input_entry_area_population.get())]}
        #print(x_test)
        test_df = pd.DataFrame(x_test)
        return self.model_obj.get_predictions(test_df)

    def set_prediction_text(self,pred_text):
        self.pred_text.config(state = tk.NORMAL)
        self.pred_text.delete(1.0,tk.END)
        self.pred_text.insert(tk.END,pred_text)
        self.pred_text.config(state = tk.DISABLED)

    def compute_and_set_pred(self):
        y_test = self.compute_prediction()
        print(y_test)
        print(type(y_test[0][0]))
        self.set_prediction_text(y_test[0][0])

    def generate_query_response(self):
        query = self.gpt_input_entry.get()
        return self.openai_obj.generate_openai_response_for_query(query)

    def set_query_resp_text(self,resp_text):
        self.gpt_text.config(state = tk.NORMAL)
        self.gpt_text.delete(1.0,tk.END)
        self.gpt_text.insert(tk.END,resp_text)
        self.gpt_text.config(state = tk.DISABLED)
        
    def generate_and_set_query_resp(self):
        resp_text = self.generate_query_response()
        sef.set_query_resp_text(resp_text)
        
    def startUI(self):
        self.root.mainloop()


# In[ ]:


class OpenAI_prompt:

    def __init__(self):
        self.client = OpenAI()

    def generate_openai_response_for_query(self,query):
        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant for answering queries from an application which demonstrates Logistic Regression. The application code is at and dataset on which Linear Regression model is trained is."},
            {"role": "user", "content": query}
        ] 
        )
        result  = completion.choices[0].message
        return result


# In[153]:


class Regression:

    def __init__(self):
        self.model = LinearRegression()
        self.is_model_trained = False

    def preprocess_data(self,df):
        df.drop(["Address"],axis=1,inplace=True)
        self.data = df
        return df.iloc[:,:-1], df.iloc[:,-1:]
        #no nulls are there so no need replace nulls .fillna(0), dropna(),
        
    def do_dataset_split(self,X,y):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)
        return X_train, X_test, y_train, y_test
        
    def train_model(self,x_train,y_train):
        self.X_train = x_train
        self.Y_train = y_train
        self.model.fit(x_train,y_train)
        self.is_model_trained = True
        print('Model Trained')

    def get_predictions(self,x_test):
        if not self.is_model_trained:
            return None
        return self.model.predict(x_test)

    def get_metrics(self,y_test,y_pred):
        if not self.is_model_trained:
            return None
        mse = mean_squared_error(y_pred,y_test)
        rms = sqrt(mse)
        res = {"mse":mse,"rms":rms}
        return res

    def plot_displot(self,col_name):
        sns.distplot(self.data[col_name])

    def plot_displot(self):
        sns.heatmap(self.data,annot=True)


# In[130]:


def add_gpt_widgets(widgets,ui_obj):
    str = "User prompt Input:"
    lbl_pos = [1,0]
    lbl_pad = [5,5]
    ui_obj.add_label(str,lbl_pos,lbl_pad)
    
    entry_pos = [1,1]
    gpt_entry = ui_obj.add_entry(40,entry_pos,lbl_pad)
    ui_obj.set_gpt_input_entry(gpt_entry)
    
    txt_dim = [40,15]
    txt_state = tk.DISABLED
    txt_pos=[8,0]
    colspan=2
    gpt_resp_text = ui_obj.add_text(txt_dim,txt_state,txt_pos,colspan,lbl_pad)
    ui_obj.set_gpt_resp_text(gpt_resp_text)

    widgets['gpt_entry'] = gpt_entry
    widgets['gpt_resp_text'] = gpt_resp_text
    


# In[172]:


def add_input_widgets_for_sample_test(widgets,ui_obj):
    padding = [5,5]
    str = ["Enter Avg. Area Income:","Enter Avg. Area House Age","Enter Avg. Area Number of Rooms","Avg. Area Number of Bedrooms","Area Population"]

    start_pos = [0,6]
    lbl_pos1 = [start_pos[0],start_pos[1]]
    ui_obj.add_label(str[0],lbl_pos1,padding)
    
    entry_pos1 = [start_pos[0],start_pos[1]+1]
    entry1 = ui_obj.add_entry(20,entry_pos1,padding)
    
    lbl_pos2 = [start_pos[0]+1,start_pos[1]]
    ui_obj.add_label(str[1],lbl_pos2,padding)
    
    entry_pos2 = [start_pos[0]+1,start_pos[1]+1]
    entry2 = ui_obj.add_entry(20,entry_pos2,padding)

    lbl_pos3 = [start_pos[0]+2,start_pos[1]]
    ui_obj.add_label(str[2],lbl_pos3,padding)
    
    entry_pos3 = [start_pos[0]+2,start_pos[1]+1] 
    entry3 = ui_obj.add_entry(20,entry_pos3,padding)

    lbl_pos4 = [start_pos[0]+3,start_pos[1]]
    ui_obj.add_label(str[3],lbl_pos4,padding)
    
    entry_pos4 = [start_pos[0]+3,start_pos[1]+1]
    entry4 = ui_obj.add_entry(20,entry_pos4,padding)

    lbl_pos5 = [start_pos[0]+4,start_pos[1]]
    ui_obj.add_label(str[4],lbl_pos5,padding)
    
    entry_pos5 = [start_pos[0]+4,start_pos[1]+1] 
    entry5 = ui_obj.add_entry(20,entry_pos5,padding)

    txt_dim = [20,1]
    txt_state = tk.DISABLED
    colspan=2
    #gpt_resp_text = ui_obj.add_text(txt_dim,txt_state,txt_pos,colspan,lbl_pad)
    text_pos6 = [start_pos[0]+4,start_pos[1]+2] 
    text6 = ui_obj.add_text(txt_dim,txt_state,text_pos6,colspan,padding)
    
    widgets['entry1'] = entry1
    widgets['entry2'] = entry2
    widgets['entry3'] = entry3
    widgets['entry4'] = entry4
    widgets['entry5'] = entry5
    widgets['pred_text'] = text6


# In[174]:


def add_buttons(widgets,ui_obj):
    start_pos = [7,6]
    padding = [5,5]
    colspan = 2
    pos = [10,0]
    button_text1 = "Generate Response for prompt"
    button_gpt = ui_obj.add_button(button_text1,ui_obj.generate_and_set_query_resp,pos,colspan,padding)
    
    pos2 = [start_pos[0],start_pos[1]]
    button_text2 = "Train & Analyze"
    button_train = ui_obj.add_button(button_text2,ui_obj.fetch_data_and_train,pos2,colspan,padding)

    pos3 = [start_pos[0],start_pos[1]+2]
    button_text3 = "Predict house price for the input"
    button_pred = ui_obj.add_button(button_text3,ui_obj.compute_and_set_pred,pos3,colspan,padding)
    
    widgets['button_train']=button_train
    widgets['button_pred']=button_pred
    widgets['button_gpt']=button_gpt
    


# In[3]:


os.environ["OPENAI_API_KEY"] = "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"


# In[175]:


ui_obj = UIgenerator("1500x1000","Data Analysis app for House Price Prediction")
model_obj = Regression()
#openai_obj = OpenAI_prompt()
data_path = "C:/Users/palakbhatia/Downloads/dataset/HousingDataset.csv"

widgets = dict()
#####for the gpt prompt########
add_gpt_widgets(widgets,ui_obj)

#####for the model input#########
add_input_widgets_for_sample_test(widgets,ui_obj)

ui_obj.set_model_obj(model_obj)
ui_obj.set_data_path(data_path)
ui_obj.set_pred_text(widgets['pred_text'])
entries = [widgets['entry1'],widgets['entry2'],
           widgets['entry3'],widgets['entry4'],
           widgets['entry5']]
ui_obj.set_model_entry_fields(entries)
#ui_obj.set_openai_obj(openai_obj)
add_buttons(widgets,ui_obj)  # plots to be added upon training


'''
ui_obj.add_button() #for train model and show results

ui_obj.add_button() # for reading from fields and showing predictions

ui_obj.add_button() # for GPT response to query entered
'''
ui_obj.startUI()


# In[99]:


def fetch_data_and_train(data_path, model_obj):
    if not model_obj.is_model_trained:
        df = pd.read_csv(dataset_path)
        model_obj.preprocess_data()
        model_obj.do_dataset_split()
        model_obj.train_model()


# In[156]:


widgets


# In[ ]:


#Data visualisation
#df.head()
#df.shape
#df.describe()
df.info()

