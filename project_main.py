# -*- coding: utf-8 -*-
"""
@author: Daniel Berenson
"""

#%% Import Packages
from sklearn import preprocessing, datasets
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense,Activation,Dropout
from tensorflow.keras.constraints import max_norm


import matplotlib.pyplot as plt
import seaborn as sns

vec = DictVectorizer()



#%% Read CSVs
fullcards = pd.read_csv('C:/Users/dfber/seaborn-data/cards.csv')
kws = pd.read_json('C:/Users/dfber/seaborn-data/Keywords.json').data.abilityWords
cardtypes = pd.read_json('C:/Users/dfber/seaborn-data/CardTypes.json').data.creature

#%% Function to extract useful data out of dataframe
def massage_creature_data (creatures):
    creatures.loc[:,"power"] = pd.to_numeric(creatures["power"], errors='coerce')
    creatures.loc[:,"toughness"] = pd.to_numeric(creatures["toughness"], errors='coerce')
    creatures.loc[:,"convertedManaCost"] = pd.to_numeric(creatures["convertedManaCost"],errors='coerce')
    creatures.power.fillna(0,inplace=True)
    creatures.toughness.fillna(0,inplace=True)
    creatures.keywords.fillna("",inplace=True)
    creatures.colorIdentity.fillna('C',inplace=True)
    creatures['rarity'] = creatures['rarity'].map({'common':0,'uncommon':1,'rare':2,'mythic':3})
    
    # Convert keywords, subtypes, and supertypes to binary variables
    for kw in kws:
       creatures["kw_" + kw] = np.where(creatures.loc[:,"keywords"].str.contains(kw),1,0)
    
    for sub in cardtypes['subTypes']:
        creatures['sub_' + sub] = np.where(creatures.loc[:,"subtypes"].str.contains(sub),1,0)
        
    for sup in cardtypes['superTypes']:
        creatures['super_' + sup] = np.where(creatures.loc[:,"subtypes"].str.contains(sup),1,0)
       
    ##Two differnet ways of converting multifarious categories into integers.
    ##This could be useful for predicting categorical Y values, but not for processing X.
    #creatures.loc[:,"keywords"] = creatures.loc[:,"keywords"].astype('category')
    #creatures["keyword_cat"] = creatures.loc[:,"keywords"].cat.codes
    #le = preprocessing.LabelEncoder()
    #creatures["colorCodes"] = le.fit_transform(creatures.loc[:,"colorIdentity"])
    
    
    creatures.sort_index(axis=0, inplace=True)
    creatures.sort_index(axis=1, inplace=True)
    
    for c in 'WUBRG':
        creatures['is' + c] = np.where(creatures.loc[:,'colorIdentity'].str.contains(c),1,0)
    
    creatures["text"] = creatures["text"].replace(np.nan,'',regex=True)
    creatures["text_length"] = creatures['text'].apply(lambda text: len(text.split()))
    
    creatures = creatures.drop(["colorIdentity","subtypes","supertypes","keywords","text"],axis=1)
    creatures = creatures.reindex(sorted(creatures.columns),axis=1)
    return creatures

#%% Feature Extraction

# Isolate the columns with useful data
usefuldata = fullcards[["borderColor","name","power","toughness","colorIdentity","types","subtypes","supertypes","convertedManaCost","keywords","text","rarity"]]
usefuldata.drop_duplicates(subset = ["name"], inplace=True)
usefuldata.set_index("name",inplace=True)
creaturecards = usefuldata[usefuldata["types"] == "Creature"].drop("types",axis=1)
creaturecards = creaturecards[(creaturecards["borderColor"] == "black") | (creaturecards["borderColor"] == "white")].drop("borderColor",axis=1)

# Massage creature data
creatures_processed = massage_creature_data(creaturecards.copy())

# To predict convertedManaCost
X_cmc = creatures_processed.drop('convertedManaCost',axis=1).values
y_cmc = creatures_processed['convertedManaCost'].values

# To predict color
X_color = creatures_processed.drop(['isW','isU','isB','isR','isG'],axis=1).values
y_color = creatures_processed[['isW','isU','isB','isR','isG']].values

#%% Train/Test Split using 60-20-20 split given total 10k samples

X_cmc_train, X_cmc_devtest, y_cmc_train, y_cmc_devtest = train_test_split(X_cmc,y_cmc,test_size=0.4,random_state=1)
X_cmc_dev, X_cmc_test, y_cmc_dev, y_cmc_test = train_test_split(X_cmc_devtest,y_cmc_devtest,test_size=0.5,random_state=1)

X_color_train, X_color_devtest, y_color_train, y_color_devtest = train_test_split(X_color,y_color,test_size=0.4,random_state=1)
X_color_dev, X_color_test, y_color_dev, y_color_test = train_test_split(X_color_devtest,y_color_devtest,test_size=0.5,random_state=1)

#%% Normalize data

cmc_scaler = preprocessing.MinMaxScaler()
X_cmc_train = cmc_scaler.fit_transform(X_cmc_train)
X_cmc_dev = cmc_scaler.transform(X_cmc_dev)
X_cmc_test = cmc_scaler.transform(X_cmc_test)

color_scaler = preprocessing.MinMaxScaler()
X_color_train = color_scaler.fit_transform(X_color_train)
X_color_dev = color_scaler.transform(X_color_dev)
X_color_test = color_scaler.transform(X_color_test)

#%% Initialize CMC TensorFlow model
cmc_model = Sequential()

# Input layer
cmc_model.add(Input(X_cmc.shape[1]))

# Hidden layers, sizes chosen arbitrarily
cmc_model.add(Dense(64,activation='relu'))
cmc_model.add(Dense(32,activation='relu'))
cmc_model.add(Dense(16,activation='relu'))

#Output layer using ReLu activation since output should be non-negative linear
cmc_model.add(Dense(1,activation='relu'))

# Compile cmc_model
cmc_model.compile(optimizer='adam',loss='MSE')

#%% Train cmc_model
cmc_model.fit(x=X_cmc_train, y=y_cmc_train, epochs=50, batch_size=256, validation_data=(X_cmc_dev,y_cmc_dev))

#%% Evaluate cmc_model

cmc_losses = pd.DataFrame(cmc_model.history.history)
cmc_losses[['loss','val_loss']].plot()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("CMC Loss")
plt.savefig("C:/Users/dfber/Google Drive/Classes/CS 230/cmc_losses.png")

cmc_predictions = cmc_model.predict(X_cmc_dev)
plt.figure()
plt.title("Mana costs")
plt.xlabel('True mana cost')
plt.ylabel('Predicted mana cost')
plt.scatter(x=y_cmc_dev,y=cmc_predictions)
plt.savefig("C:/Users/dfber/Google Drive/Classes/CS 230/mana_costs.png")


#%% Initialize Color TensorFlow model
color_model = Sequential()

# Input layer
color_model.add(Input(X_color.shape[1]))

# Hidden layers, sizes chosen arbitrarily
color_model.add(Dense(64,activation='relu'))
color_model.add(Dense(32,activation='relu'))
color_model.add(Dense(16,activation='relu'))

#Output layer using ReLu activation since output should be non-negative linear
color_model.add(Dense(5,activation='sigmoid'))

# Compile color_model
color_model.compile(optimizer='adam',loss='binary_crossentropy')


#%% Train color_model
color_model.fit(x=X_color_train, y=y_color_train, epochs=50, batch_size=256, validation_data=(X_color_dev,y_color_dev))

#%% Evaluate color_model

color_losses = pd.DataFrame(color_model.history.history)
color_losses[['loss','val_loss']].plot()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Color Loss")
plt.savefig("C:/Users/dfber/Google Drive/Classes/CS 230/color_losses.png")

color_predictions = color_model.predict(X_color_dev)
color_predictions_binarized = color_predictions > 0.5

print("White:")
print(classification_report(y_color_dev[:,0],color_predictions_binarized[:,0]))
print("Blue:")
print(classification_report(y_color_dev[:,1],color_predictions_binarized[:,1]))
print("Black:")
print(classification_report(y_color_dev[:,2],color_predictions_binarized[:,2]))
print("Red:")
print(classification_report(y_color_dev[:,3],color_predictions_binarized[:,3]))
print("Green:")
print(classification_report(y_color_dev[:,4],color_predictions_binarized[:,4]))

#%% Assess model on my own idea for a a creature

def color_predictions_to_identity(prediction_array):
    identity = ""
    if prediction_array[:,0] > 0.5:
        identity += 'W'
    if prediction_array[:,1] > 0.5:
        identity += 'U'
    if prediction_array[:,2] > 0.5:
        identity += 'B'
    if prediction_array[:,3] > 0.5:
        identity += 'R'
    if prediction_array[:,4] > 0.5:
        identity += 'G'
    return identity


#creature_name = "Squee, Goblin Nabob"
#creature_name = "Snapcaster Mage"
#creature_name = "Serra Angel"
creature_name = "Ruric Thar, the Unbowed"

example_creature = creaturecards.loc[creature_name].to_frame().transpose()
example_creature_processed = massage_creature_data(example_creature.copy())
example_creature_X_cmc = cmc_scaler.transform(example_creature_processed.drop('convertedManaCost',axis=1).values)
example_creature_cmc_prediction = cmc_model.predict(example_creature_X_cmc)
example_creature_X_color = color_scaler.transform(example_creature_processed.drop(['isW','isU','isB','isR','isG'],axis=1).values)
example_creature_color_prediction = color_model.predict(example_creature_X_color)

print(creature_name + " is predicted to have CMC " + str(example_creature_cmc_prediction) + ", and its true CMC is " + str(example_creature["convertedManaCost"].values))
print(creature_name + " is predicted to have color identity " + color_predictions_to_identity(example_creature_color_prediction) + ", and its true color identity is " + example_creature["colorIdentity"].values)


my_creature = pd.DataFrame(index = ["my_creature"], data = {
        'power': 2,
        'toughness': 2,
        'colorIdentity': 'U',
        'subtypes': 'Wizard',
        'supertypes': "",
        'convertedManaCost': 2,
        'keywords': "",
        'text': "",
        'rarity': "common"
        })
my_creature_processed = massage_creature_data(my_creature.copy())
my_creature_X_cmc = cmc_scaler.transform(my_creature_processed.drop('convertedManaCost',axis=1).values)
my_creature_cmc_prediction = cmc_model.predict(my_creature_X_cmc)
my_creature_X_color = color_scaler.transform(my_creature_processed.drop(['isW','isU','isB','isR','isG'],axis=1).values)
my_creature_color_prediction = color_model.predict(my_creature_X_color)
print("My creature is predicted to have CMC " + str(my_creature_cmc_prediction))
print("My creature is predicted to have color identity " + color_predictions_to_identity(my_creature_color_prediction))


#%%
#plt.figure()
#sns.scatterplot(x='power',y='toughness',hue='CMC>3',data=creatures)
#plt.figure()
#sns.scatterplot(x='power',y='convertedManaCost',hue='colorIdentity',data=creatures.loc[creatures['colorIdentity'].apply(lambda c: c in 'WUBRG')])
#plt.figure()
#sns.scatterplot(x='power',y='convertedManaCost',data=creatures)
#plt.figure()
#sns.stripplot(x='power',y='convertedManaCost',data=creatures,jitter=True)
#plt.figure()
#sns.kdeplot(x='power',y='convertedManaCost',data=creatures)
#plt.figure()
#sns.scatterplot(x='power',y='toughness',hue='convertedManaCost',data=creatures)
#plt.figure()
#sns.violinplot(x='colorIdentity',y='power',hue='CMC>3',data=creatures.loc[creatures['colorIdentity'].apply(lambda c: c in 'WUBRG')])
#plt.figure()
#sns.violinplot(x='isW',y='power',hue='CMC>3',data=creatures)