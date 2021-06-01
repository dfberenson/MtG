# -*- coding: utf-8 -*-
"""
Created on Fri May 28 22:21:02 2021

@author: dfber
"""

#%% Import Packages
from sklearn import preprocessing, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report, f1_score
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.spatial.distance import cosine
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Input,Dense,Activation,Dropout,Embedding,LSTM,GRU,Concatenate
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import datetime

from random import choice
import matplotlib.pyplot as plt
import seaborn as sns


#%% Read CSVs

start_datetime = datetime.datetime.now()

plt.close('all')

fullcards = pd.read_csv('C:/Users/dfber/seaborn-data/cards.csv')
kws = pd.read_json('C:/Users/dfber/seaborn-data/Keywords.json').data.abilityWords
cardtypes = pd.read_json('C:/Users/dfber/seaborn-data/CardTypes.json').data.creature


#%% Read Glove vectors using code from Coursera

with open('C:/Users/dfber/gensim-data/glove.6B.50d.txt', 'r', encoding='utf8') as f:
    words = set()
    word_to_vec_map = {}
    
    for line in f:
        line = line.strip().split()
        curr_word = line[0]
        words.add(curr_word)
        word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

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
    
    # Replace creatures' own names in their textboxes with 'CARDNAME' as is done by designers
    for creat in creatures.index:
        creatures.loc[creat,"text"] = creatures.loc[creat,"text"].replace(creat,"CARDNAME")
    
    creatures = creatures.drop(["colorIdentity","subtypes","supertypes","keywords"],axis=1)
    creatures = creatures.reindex(sorted(creatures.columns),axis=1)
    return creatures


def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similarity between u and v
        
    Arguments:
        u -- a word vector of shape (n,)          
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """
        
    ### START CODE HERE ###
    # Compute the dot product between u and v (≈1 line)
    dot = np.dot(u,v)
    # Compute the L2 norm of u (≈1 line)
    norm_u = np.sqrt(np.sum(u**2))
    
    # Compute the L2 norm of v (≈1 line)
    norm_v = np.sqrt(np.sum(v**2))
    # Compute the cosine similarity defined by formula (1) (≈1 line)
    cosine_similarity = dot / (norm_u*norm_v)
    ### END CODE HERE ###
    
    return cosine_similarity

#%% Feature Extraction

# Isolate the columns with useful data
usefuldata = fullcards[["borderColor","name","power","toughness","colorIdentity","types","subtypes","supertypes","convertedManaCost","keywords","text","rarity"]]
usefuldata.drop_duplicates(subset = ["name"], inplace=True)
usefuldata.set_index("name",inplace=True)
creaturecards = usefuldata[usefuldata["types"] == "Creature"].drop("types",axis=1)
creaturecards = creaturecards[(creaturecards["borderColor"] == "black") | (creaturecards["borderColor"] == "white")].drop("borderColor",axis=1)

# Massage creature data
creatures_processed = massage_creature_data(creaturecards.copy())

X_words = creatures_processed['text'].values.astype('str')

# To predict convertedManaCost
X_cmc = creatures_processed.drop(['convertedManaCost','text'],axis=1).values
y_cmc = creatures_processed['convertedManaCost'].values

# To predict color
X_color = creatures_processed.drop(['isW','isU','isB','isR','isG','text'],axis=1).values
y_color = creatures_processed[['isW','isU','isB','isR','isG']].values


#%% Create tensorflow embedding
# Following https://keras.io/examples/nlp/pretrained_word_embeddings/

vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=200)
text_ds = tf.data.Dataset.from_tensor_slices(creatures_processed["text"].values.astype('str')).batch(128)
vectorizer.adapt(text_ds)
voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))

num_tokens = len(voc) + 2
embedding_dim = 50
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = word_to_vec_map.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        print("Missed " + word)
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses)) # There are lots of uncommon and invented words so many misses



#%% Train/Test Split using 60-20-20 split given total 10k samples

X_cmc_words_train, X_cmc_words_devtest, y_cmc_train, y_cmc_devtest = train_test_split(X_words,y_cmc,test_size=0.4,random_state=1)
X_cmc_words_dev, X_cmc_words_test, y_cmc_dev, y_cmc_test = train_test_split(X_cmc_words_devtest,y_cmc_devtest,test_size=0.5,random_state=1)

X_color_words_train, X_color_words_devtest, y_color_words_train, y_color_words_devtest = train_test_split(X_words,y_color,test_size=0.4,random_state=1)
X_color_words_dev, X_color_words_test, y_color_words_dev, y_color_words_test = train_test_split(X_color_words_devtest,y_color_words_devtest,test_size=0.5,random_state=1)

#%% Convert the text data into their integer indices using the vectorizer

X_cmc_words_train = vectorizer(np.array([[s] for s in X_cmc_words_train])).numpy()
X_cmc_words_dev = vectorizer(np.array([[s] for s in X_cmc_words_dev])).numpy()
X_cmc_words_test = vectorizer(np.array([[s] for s in X_cmc_words_test])).numpy()

X_color_words_train = vectorizer(np.array([[s] for s in X_color_words_train])).numpy()
X_color_words_dev = vectorizer(np.array([[s] for s in X_color_words_dev])).numpy()
X_color_words_test = vectorizer(np.array([[s] for s in X_color_words_test])).numpy()

#%% Set hyperparameters

hidden_dense_layers = 2
initial_dense_layer_size = 128
use_dropout = True
dropout_rate = 0.2
epochs = 28

#%% Initialize CMC TensorFlow model
cmc_model = Sequential()

# Input layer
cmc_model.add(Input((None,)))

cmc_model.add(Embedding(
    input_dim = num_tokens,
    output_dim = embedding_dim,
    embeddings_initializer=Constant(embedding_matrix),
    trainable=True, # Want to be able to train the embeddings for unusual words like "treefolk"
))

cmc_model.add(LSTM(128))

# Hidden layers, sizes decreasing powers of 2
for j in range(hidden_dense_layers):
    cmc_model.add(Dense(initial_dense_layer_size / 2 ** j, activation='relu'))
    if use_dropout:
        cmc_model.add(Dropout(dropout_rate))

#Output layer using ReLu activation since output should be non-negative linear
cmc_model.add(Dense(1,activation='relu'))

# Compile cmc_model
cmc_model.compile(optimizer='adam',loss='MSE')

#%% Train cmc_model
cmc_model.fit(x=X_cmc_words_train, y=y_cmc_train, epochs=epochs, batch_size=256, validation_data=(X_cmc_words_dev,y_cmc_dev))

#%% Initialize Color TensorFlow model
color_model = Sequential()

# Input layer
color_model.add(Input((None,)))

color_model.add(Embedding(
    input_dim = num_tokens,
    output_dim = embedding_dim,
    embeddings_initializer=Constant(embedding_matrix),
    trainable=True, # Want to be able to train the embeddings for unusual words like "treefolk"
))

color_model.add(LSTM(128))

# Hidden layers, sizes decreasing powers of 2
for j in range(hidden_dense_layers):
    color_model.add(Dense(initial_dense_layer_size / 2 ** j, activation='relu'))
    if use_dropout:
        color_model.add(Dropout(dropout_rate))

#Output layer using ReLu activation since output should be non-negative linear
color_model.add(Dense(5,activation='sigmoid'))

# Compile color_model
color_model.compile(optimizer='adam',loss='binary_crossentropy')


#%% Train color_model
color_model.fit(x=X_color_words_train, y=y_color_words_train, epochs=epochs, batch_size=256, validation_data=(X_color_words_dev,y_color_words_dev))

#%% Evaluate cmc_model

cmc_losses = pd.DataFrame(cmc_model.history.history)
cmc_losses[['loss','val_loss']].plot()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("CMC Loss")
plt.savefig("C:/Users/dfber/Google Drive/Non-med Classes/CS 230/cmc_losses.png")

cmc_predictions = cmc_model.predict(X_cmc_words_dev)
cmc_dev_df = pd.DataFrame(data={"True": y_cmc_dev, "Predicted": cmc_predictions.ravel()})
plt.figure()
plt.title("Mana costs")
plt.xlabel('True mana cost')
plt.ylabel('Predicted mana cost')
sns.violinplot(x = "True", y = "Predicted", data = cmc_dev_df)
plt.savefig("C:/Users/dfber/Google Drive/Non-med Classes/CS 230/mana_costs.png")

print("CMC model mean squared error is " + str(mean_squared_error(y_cmc_dev, cmc_model.predict(X_cmc_words_dev))))

#%% Evaluate color_model

color_losses = pd.DataFrame(color_model.history.history)
color_losses[['loss','val_loss']].plot()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Color Loss")
plt.savefig("C:/Users/dfber/Google Drive/Non-med Classes/CS 230/color_losses.png")

color_predictions = color_model.predict(X_color_words_dev)
color_predictions_binarized = color_predictions > 0.5

print("White:")
print(classification_report(y_color_words_dev[:,0],color_predictions_binarized[:,0]))
print("Blue:")
print(classification_report(y_color_words_dev[:,1],color_predictions_binarized[:,1]))
print("Black:")
print(classification_report(y_color_words_dev[:,2],color_predictions_binarized[:,2]))
print("Red:")
print(classification_report(y_color_words_dev[:,3],color_predictions_binarized[:,3]))
print("Green:")
print(classification_report(y_color_words_dev[:,4],color_predictions_binarized[:,4]))




#%% Attempt to combine NLP models with structured data from original model.
# The textual data is embedded, run through an LSTM layer and then a dense layer,
# and then these outputs are concatenated with the raw structured data from the
# main part of the project.
# I think this general idea would work and be quite powerful, but I was not able
# to figure out the code required

combine_NLP_and_structured_data = False

if combine_NLP_and_structured_data:

    X_nlp = Input((200,50))
    X_nlp = Embedding(
        input_dim = num_tokens,
        output_dim = embedding_dim,
        embeddings_initializer=Constant(embedding_matrix),
        trainable=True, # Want to be able to train the embeddings for unusual words like "treefolk"
    )(X_nlp)
    X_nlp = LSTM(128)(X_nlp)
    NLP_outputs = Dense(64,activation='relu')(X_nlp)
    
    X_structured = Input(X_cmc.shape[1])
    
    X_final = Concatenate([NLP_outputs,X_structured])
    
    X_final = Dense(128)(X_final)
    X_final = Dropout(0.2)(X_final)
    X_final = Dense(64)(X_final)
    X_final = Dropout(0.2)(X_final)
    
    combined_cmc_model = Model(inputs = [X_nlp,X_structured], outputs = X_final)
    
    combined_cmc_model.compile(optimizer='adam',loss='MSE')
    combined_cmc_model.fit(x=[X_cmc_train,X_cmc_words_train], y=y_color_train, epochs=epochs, batch_size=256, validation_data=(X_color_dev,y_color_dev))
        # And similarly for color
