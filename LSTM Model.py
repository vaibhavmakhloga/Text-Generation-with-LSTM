def create_model(vocabulary_size,seq_len):

    model=Sequential()
    model.add(Embedding(vocabulary_size,seq_len,input_length=seq_len))
    model.add(LSTM(seq_len*2,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(50,activation='relu'))

    model.add(Dense(vocabulary_size,activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    model.summary()
    return model
