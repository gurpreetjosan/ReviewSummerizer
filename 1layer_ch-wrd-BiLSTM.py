import glob
import os
import re
from pickle import dump

import numpy as np
from keras.callbacks import ModelCheckpoint,EarlyStopping
from numpy import array
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model,np_utils
from keras.models import Model
from keras.layers import Input, Bidirectional, TimeDistributed, Dropout, Masking, concatenate, Reshape
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
import xml.etree.ElementTree as ET
from os import listdir
from keras.optimizers import SGD

from sklearn.preprocessing import LabelEncoder
outFileName="1L_ch-wrd-BiLSTM" # for storing all files related to this model



# load doc into memory
def load_doc(filename):
    # print("load_doc")
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def create_tokenizer(descriptions):
    tokenizer = Tokenizer(oov_token="~",filters='')
    tokenizer.fit_on_texts(descriptions)
    return tokenizer

def load_data_and_labels():
    dic = dict()
    sents=[]
    i=0
    label = []
    path='./reviewfiles/*.txt'
    files = glob.glob(path)
    for name in files:
        with open (name) as myfile:
                for line in myfile:
                    line = str.lower(line)
                    #line = re.sub("(\\p{Alpha})\\1+","\1\1",line)
                    line = re.sub(r'(\w)\1+', r'\g<1>\g<1>', line)
                    x1,x2 = line.split(',')
                    #dic[i]=x1.strip()
                    i=i+1
                    sents.append(x1.strip())
                    label.append(x2.strip())

    return sents,label


    #return np.asarray(sents), np.asarray(labels)

# define the model

#model definition section starts here
def wordmodel(max_len):
    input = Input(shape=(max_len,))
    mask = Masking(mask_value=0)(input)
    model = Embedding(input_dim=wrd_vocab_size, output_dim=100, input_length=max_len, mask_zero=True)(mask)
    model = Dropout(0.1)(model)
    model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
    wordmodel = TimeDistributed(Dense(100))(model)  # softmax output layer
    print (wordmodel)
    return Model(input,wordmodel)

def charmodel(maxcharlen):
    input = Input(shape=(maxcharlen,))
    mask = Masking(mask_value=0)(input)
    model = Embedding(input_dim=char_vocab_size, output_dim=50, input_length=maxcharlen, mask_zero=True)(mask)
    model = Dropout(0.1)(model)
    model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
    charmodel = TimeDistributed(Dense(100))(model)  # softmax output layer
    #charmodel=Flatten()(charmodel)
    return Model(input,charmodel)


def define_model():
    wordseq = Input(shape=(max_sent_length,))
    charSeq= Input(shape=(max_sent_length,max_wrd_len))
    wm=wordmodel(max_sent_length)(wordseq)
    cm=TimeDistributed(charmodel(max_wrd_len))(charSeq)
    cm = Reshape((max_sent_length, -1))(cm)
    combined_input=concatenate([wm,cm])
    model = Bidirectional(LSTM(units=100, recurrent_dropout=0.1))(combined_input)
    out = Dense(10, activation="softmax")(model)  # softmax output layer

    model = Model([wordseq,charSeq], out)

    #load existing weight if exist
    if os.path.isfile(outFileName+"-best.hdf5"):
        model.load_weights(outFileName+"-best.hdf5")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    #plot_model(model, show_shapes=True, to_file=outFileName+'-plot.png')
    return model


# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

#get a char seq and convert into ids and pad them
def getchardata(d,flag):
    chardata=[]
    for i in range(len(d)):
        tuple=[]
        # add start token if generating prev word seq
        if flag==1:
            seq=char_tokenizer.texts_to_sequences(["~"])
            seq=pad_sequences(seq,max_wrd_len,padding='post')
            tuple.append(seq[0])
        j=0
        #convert every word of line at position i in data into seq of character and get their ids, pad them and put in tuple
        for wrd in d[i].split():
            if flag==1 and j==len(d[i])-1:  #if prev word seq, then skip last word
                break
            if flag==3 and j==0:  #if nxt word seq, skip first word
                continue
            seq=char_tokenizer.texts_to_sequences([list(wrd)])
            seq=pad_sequences(sequences=seq,maxlen=max_wrd_len,padding='post')
            tuple.append(seq[0])
            j+=1
        #----------- for loop ends here
        if flag==3:  # if nxt word seq generating then add end line token after padding
            seq=char_tokenizer.texts_to_sequences(["~"])
            seq=pad_sequences(seq,max_wrd_len,padding='post')
            tuple.append(seq[0])
        #check if length of tuple is equal to max length, if not then pad with zero
        if(len(tuple)<max_sent_length):
            while len(tuple)!= max_sent_length :
                tuple.append([0]*max_wrd_len)
        chardata.append(np.array(tuple))
    return np.array(chardata)


#----Main logic starts here

# load dev set
dataset,labels = load_data_and_labels()
print('Dataset: %d' % len(dataset))

# prepare tokenizer
tokenizer = create_tokenizer(dataset)
wrd_vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % wrd_vocab_size)
#save tokenizer
dump(tokenizer, open(outFileName+'-wrdtokenizer.pkl', 'wb'))

# determine the maximum sequence length
max_sent_length = max(len(s.split()) for s in dataset)
print('Description Length: %d' % max_sent_length)
max_wrd_len=max(len(wrd) for s in dataset for wrd in s.split())
print("max_wrd_len...",max_wrd_len)
#max_wrd_len=30
#prepare char tokenizer
str1=""
for i in range(len(dataset)):
    str1 = str1+''.join(dataset[i])
char_tokenizer=create_tokenizer(str1)
char_vocab_size = len(char_tokenizer.word_index) + 1
print('Vocabulary Size: %d' % char_vocab_size)
#save tokenizer
dump(char_tokenizer, open(outFileName+'-chartokenizer.pkl', 'wb'))


# train-test split
train_descriptions, test_descriptions, train_labels, test_labels = train_test_split(dataset,labels, test_size=0.0, random_state=42)
print('Descriptions: train=%d, test=%d' % (len(train_descriptions), len(test_descriptions)))
#dump(train_descriptions, open(outFileName+'-train_descriptions.pkl', 'wb'))
#dump(test_descriptions, open(outFileName+'-test_descriptions.pkl', 'wb'))
#dump(train_labels, open(outFileName+'-train_labels.pkl', 'wb'))
#dump(test_labels, open(outFileName+'-test_labels.pkl', 'wb'))

#get sequence of training data
trainseq= tokenizer.texts_to_sequences(train_descriptions)
trainseq = pad_sequences(trainseq, maxlen=max_sent_length,padding='post')

#get sequence of characters for training data
train_charseq=getchardata(train_descriptions,2)


#convert string labels to numeric
encoder = LabelEncoder()
encoder.fit(train_labels)

#dump(encoder, open(outFileName+'-label_encoder.pkl', 'wb'))
num_classes=encoder.classes_.size
train_labels = encoder.transform(train_labels).astype(np.int32)
out_seq = to_categorical([train_labels], num_classes=num_classes)[0]

# define experiment

# run experiment
model = define_model()
# checkpoint
filepath=outFileName+"-best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
earlystop=EarlyStopping(monitor="val_loss",patience=2)
callbacks_list = [checkpoint,earlystop]
hist = model.fit([trainseq, train_charseq], out_seq, epochs=25, batch_size=32,verbose=1, callbacks=callbacks_list,validation_split=0.1)
