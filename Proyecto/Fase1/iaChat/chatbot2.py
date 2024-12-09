import re
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Embedding, LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# Cargar los datos
lines = open('./corpus/movie_lines.txt', encoding='UTF-8', errors='ignore').read().split('\n')
convers = open('./corpus/movie_conversations.txt', encoding='UTF-8', errors='ignore').read().split('\n')

exchn = []
for conver in convers:
    exchn.append(conver.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(",", "").split())

diag = {}
for line in lines:
    line_splitted = line.split(' +++$+++ ')
    diag[line_splitted[0]] = line_splitted[-1]

questions = []
answers = []

for conver in exchn:
    for i in range(len(conver) - 1):
        questions.append(diag[conver[i]])
        answers.append(diag[conver[i + 1]])

del(conver, convers, diag, exchn, i, line, lines)

# Filtrar las preguntas y respuestas más largas que 12 palabras
sorted_ques = []
sorted_ans = []
for i in range(len(questions)):
    if len(questions[i]) < 13:
        sorted_ques.append(questions[i])
        sorted_ans.append(answers[i])

# Limpiar el texto
def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r"i'm", "i am", txt)
    txt = re.sub(r"he's", "he is", txt)
    txt = re.sub(r"she's", "she is", txt)
    txt = re.sub(r"that's", "that is", txt)
    txt = re.sub(r"what's", "what is", txt)
    txt = re.sub(r"where's", "where is", txt)
    txt = re.sub(r"\'ll", " will", txt)
    txt = re.sub(r"\'ve", " have", txt)
    txt = re.sub(r"\'re", " are", txt)
    txt = re.sub(r"\'d", " would", txt)
    txt = re.sub(r"won't", "will not", txt)
    txt = re.sub(r"can't", "can not", txt)
    txt = re.sub(r"[^\w\s]", "", txt)
    return txt

clean_ques = [clean_text(line) for line in sorted_ques]
clean_ans = [clean_text(line) for line in sorted_ans]

for i in range(len(clean_ans)):
    clean_ans[i] = ' '.join(clean_ans[i].split()[:11])

clean_ans = clean_ans[:2000]
clean_ques = clean_ques[:2000]

word2count = {}
for line in clean_ques + clean_ans:
    for word in line.split():
        word2count[word] = word2count.get(word, 0) + 1

thresh = 5
vocab = {}
word_num = 0
for word, count in word2count.items():
    if count >= thresh:
        vocab[word] = word_num
        word_num += 1

tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    vocab[token] = word_num
    word_num += 1

inv_vocab = {v: w for w, v in vocab.items()}

# Convertir las preguntas y respuestas a secuencias
encoder_inp = []
for line in clean_ques:
    lst = [vocab.get(word, vocab['<OUT>']) for word in line.split()]
    encoder_inp.append(lst)

decoder_inp = []
for line in clean_ans:
    lst = [vocab.get(word, vocab['<OUT>']) for word in line.split()]
    decoder_inp.append(lst)

encoder_inp = pad_sequences(encoder_inp, 13, padding='post', truncating='post')
decoder_inp = pad_sequences(decoder_inp, 13, padding='post', truncating='post')

decoder_final_output = []
for i in decoder_inp:
    decoder_final_output.append(i[1:])
decoder_final_output = pad_sequences(decoder_final_output, 13, padding='post', truncating='post')

decoder_final_output = to_categorical(decoder_final_output, len(vocab))

# Crear el modelo Seq2Seq sin atención
enc_inp = Input(shape=(13,))
dec_inp = Input(shape=(13,))

VOCAB_SIZE = len(vocab)
embed = Embedding(VOCAB_SIZE + 1, output_dim=50, input_length=13, trainable=True)

# Codificador
enc_embed = embed(enc_inp)
enc_lstm = LSTM(512, return_sequences=True, return_state=True)  # Aumentar unidades LSTM
enc_op, h, c = enc_lstm(enc_embed)
enc_states = [h, c]

# Decodificador con LSTM
dec_embed = embed(dec_inp)
dec_lstm = LSTM(512, return_sequences=True, return_state=True)
dec_op, _, _ = dec_lstm(dec_embed, initial_state=enc_states)

# Capa densa de salida
dense = Dense(VOCAB_SIZE, activation='softmax')
dense_op = dense(dec_op)

# Modelo final
model = Model([enc_inp, dec_inp], dense_op)
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')

# Entrenar el modelo
model.fit([encoder_inp, decoder_inp], decoder_final_output, epochs=5)

# Guardar el modelo
model.save('seq2seq_model_no_attention.h5')

# -----------------------------------------------------------------------------------------------------------
# Definir el proceso de predicción
model = load_model('seq2seq_model_no_attention.h5')

encoder_model = Model(enc_inp, enc_states)
decoder_model = Model([dec_inp] + enc_states, dense_op)

def predict_response(input_text):
    input_text = clean_text(input_text)
    input_sequence = [vocab.get(word, vocab['<OUT>']) for word in input_text.split()]
    input_sequence = pad_sequences([input_sequence], maxlen=13, padding='post', truncating='post')

    # Codificar la entrada
    encoder_states = encoder_model.predict(input_sequence)

    # Comenzar con el token <SOS>
    target_sequence = [vocab['<SOS>']]

    # Generar la respuesta palabra por palabra
    stop_condition = False
    output_text = ''
    while not stop_condition:
        target_sequence_input = pad_sequences([target_sequence], maxlen=13, padding='post', truncating='post')
        output_tokens = decoder_model.predict([target_sequence_input] + encoder_states)

        # Obtener la palabra con la probabilidad más alta
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = inv_vocab.get(sampled_token_index, '<OUT>')

        # Agregar la palabra a la respuesta
        output_text += ' ' + sampled_word

        # Detener si se genera <EOS>
        if sampled_word == '<EOS>' or len(output_text.split()) > 12:
            stop_condition = True

        target_sequence.append(sampled_token_index)

    return output_text.strip()

# -----------------------------------------------------------------------------------------------------------
print("¡Hola! Soy un chatbot. Puedes empezar a hablarme y te responderé. (Escribe 'salir' para terminar)")

while True:
    input_text = input("Tú: ")
    if input_text.lower() == 'salir':
        print("Chatbot: ¡Adiós!")
        break

    response = predict_response(input_text)

    print(f"Chatbot: {response}")
