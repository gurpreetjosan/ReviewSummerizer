import glob
import errno
from pickle import dump

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from numpy import array
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
import numpy as np
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Lambda, Dropout, Bidirectional
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import model_from_json
from keras.utils import plot_model
print ("new 2 layer...!!!!!!")

outfilename="2L_LSTM"
def create_tokenizer(desc):
    lines = list(desc.values())
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer
def create_lines():
	dic = dict()
	list1=[]
	i=0
	list2 = []
	path='./reviewfiles/*.txt'
	files = glob.glob(path)
	for name in files:
		with open (name) as myfile:
				for line in myfile:
					x1,x2 = line.split(',')

					dic[i]=line
					i=i+1
					list1.append(x1.strip())
					list2.append(x2.strip())

	return dic,list1,list2
def count_word():
	count=0
	max=0
	dic, lines,label = create_lines()
	for line in lines:
		for i in line.split(" "):
			count=count+1
			if(count>max):
				max=count
		count=0
	return max

# define the model multiplelayers
def define_model(vocab_size, max_length):
    # embedding
    inputs2 = Input(shape=(max_length,))
    emb2 = Embedding(vocab_size,  300, mask_zero=True)(inputs2)
    emb2=Dropout(0.3)(emb2)
    emb3 = LSTM(300,return_sequences=True)(emb2)
#    fc = TimeDistributed(Dense(250, activation='relu'))(emb3)
    emb3 = Dropout(0.3)(emb3)
    lm2 = LSTM(300)(emb3)
    lm2=Dropout(0.3)(lm2)
    outputs = Dense(10, activation='softmax')(lm2)
    # tie it together
    model = Model(inputs=[inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
   # plot_model(model, show_shapes=True, to_file=outfilename+'plot.png')
    return model


dic,dataset,labels=create_lines()
tokenizer=create_tokenizer(dic)
#dump(tokenizer, open(outfilename+'-tokenizer.pkl', 'wb'))
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)
max_length=count_word()
print("----")
train_data, test_data,train_labels,test_labels = train_test_split(dataset,labels, test_size=0.1, random_state=42)
print('Descriptions: train=%d, test=%d' % (len(train_data), len(test_data)))
model_name = outfilename
verbose = 1
#n_epochs = 10

trainseq= tokenizer.texts_to_sequences(train_data)
trainseq = pad_sequences(trainseq, maxlen=max_length)

#convert string labels to numeric
encoder = LabelEncoder()
encoder.fit(train_labels)

num_classes=encoder.classes_.size
train_labels = encoder.transform(train_labels).astype(np.int32)
out_seq = to_categorical([train_labels], num_classes)[0]

# define experimentses

# run experiment
model = define_model(vocab_size, max_length)
# checkpoint
filepath=outfilename+"-best.hdf5"
earlystopping=keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=2,verbose=0, mode='auto')
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint,earlystopping]
hist = model.fit(trainseq, out_seq, epochs=50, batch_size=32,verbose=1, callbacks=callbacks_list,validation_split=0.1)

