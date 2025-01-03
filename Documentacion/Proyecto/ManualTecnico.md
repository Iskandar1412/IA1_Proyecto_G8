# Manual Técnico

## Introducción

En el desarrollo del proyecto en su fase final de Inteligencia Artificial 1, se implementó una solución utilizando python y Tikter como tecnologías principales para construir una interfaz gráfica interactiva que permita la comunicación con un modelo de inteligencia artificial tanto para idioma español, ingles e incluso código (generador de código). Este modelo, inicialmente desarrollado en Python, trasladando y entrenando el modelo para que fuese de tipo tensorflow para su implementación en la interfáz.

La funcionalidad principal del sistema consiste en recibir entradas de texto en español/ingles, procesarlas a través de un modelo previamente entrenado y proporcionar respuestas coherentes en tiempo real (aplicando para código, lo cual representa para esta fase del proyecto <fase 3>). Para este propósito, se emplearon herramientas de conversión y serialización para traducir el modelo desarrollado en Python.

El enfoque del proyecto incluyó la creación de una estructura modular en Python, lo que facilitó la gestión de componentes y el manejo del estado, mientras que el uso de Tkinter para el se utilizó para el desarrollo del entorno virtual, conectando la interfaz gráfica con el modelo, permitiendo una interacción eficiente y de fácil uso.

Este proyecto demostró la capacidad de implementar un modelo de inteligencia artificial funcional para entornos de escritorio, asegurando un alto nivel de rendimiento y accesibilidad para los usuarios.

## Requisitos Mínimos del Sistema

> **Systema Operativo:** Windows 7 o superior, Ubuntu 22.04 o superior, arch linux
> **CPU:** Intel Pentium D o AMD Athlon 64 (K8) 2.6GHz o superior
> **RAM:** 8GB
> **Lenguaje Modelo:** Tkinter
> **Lenguaje Modelo:** Python (version 3.12.8)
> **IDE:** Visual Studio Code

## Explicación del Código

###  ModeloChatA.ipynb

```py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import pandas as pd

# ============================
# Configuración de parámetros
# ============================
latent_dim = 256  # Dimensión del espacio latente para LSTM
num_samples = 10000  # Número máximo de muestras
max_input_length = 20  # Longitud máxima de las secuencias de entrada
max_output_length = 20  # Longitud máxima de las secuencias de salida


# Leer el archivo TSV en español con pandas
file_path_es = "dataset_es.tsv"
data_es = pd.read_csv(file_path_es, sep="\t")

# Leer el archivo TSV en ingles con pandas
file_path_en = "dataset_en.tsv"
data_en = pd.read_csv(file_path_en, sep="\t")

# Extraer columnas "Question" y "Answer" del dataset en español
input_texts_es = data_es["Question"].tolist()
output_texts_es = ["<start> " + answer_es + " <end>" for answer_es in data_es["Answer"].tolist()]

# Extraer columnas "Question" y "Answer" del dataset en inglés
input_texts_en = data_en["Question"].tolist()
output_texts_en = ["<start> " + answer_en + " <end>" for answer_en in data_en["Answer"].tolist()]

# Unificación de dataset en inglés y español
input_texts = input_texts_es + input_texts_en
output_texts = output_texts_es + output_texts_en
# ========================
# Preprocesamiento de datos
# ========================
# Tokenización de las secuencias
input_tokenizer = Tokenizer()
output_tokenizer = Tokenizer(filters="")  # Importante: no filtrar caracteres especiales

input_tokenizer.fit_on_texts(input_texts)
output_tokenizer.fit_on_texts(output_texts)

input_sequences = input_tokenizer.texts_to_sequences(input_texts)
output_sequences = output_tokenizer.texts_to_sequences(output_texts)

# Agregar padding para las secuencias
encoder_input_data = pad_sequences(input_sequences, maxlen=max_input_length, padding="post")
decoder_input_data = pad_sequences([seq[:-1] for seq in output_sequences], maxlen=max_output_length, padding="post")
decoder_target_data = pad_sequences([seq[1:] for seq in output_sequences], maxlen=max_output_length, padding="post")

# Convertir los objetivos en one-hot encoding
num_decoder_tokens = len(output_tokenizer.word_index) + 1
decoder_target_data = tf.keras.utils.to_categorical(decoder_target_data, num_decoder_tokens)

# ===================
# Construcción del modelo
# ===================
# Encoder
encoder_inputs = Input(shape=(None,), dtype="int32")  # Entrada es una secuencia de índices
encoder_embedding = tf.keras.layers.Embedding(input_dim=len(input_tokenizer.word_index) + 1,
                                               output_dim=latent_dim,
                                               mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,), dtype="int32")  # Entrada es una secuencia de índices
decoder_embedding = tf.keras.layers.Embedding(input_dim=len(output_tokenizer.word_index) + 1,
                                               output_dim=latent_dim,
                                               mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(len(output_tokenizer.word_index) + 1, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

# Modelo Seq2Seq
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compilar el modelo
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Resumen del modelo
model.summary()

# ===================
# Entrenamiento del modelo
# ===================
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=16,
    epochs=50,
    validation_split=0.2,
    callbacks=[
        EarlyStopping(patience=5, monitor="val_loss"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
    ]
)

# ===================
# Modelos para inferencia
# ===================
# Encoder para inferencia
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder para inferencia
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_lstm_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_lstm_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states,
)

# ===================
# Función para decodificar secuencias
# ===================
reverse_input_word_index = dict((i, word) for word, i in input_tokenizer.word_index.items())
reverse_output_word_index = dict((i, word) for word, i in output_tokenizer.word_index.items())


def decode_sequence(input_seq):
    # Obtener los estados del encoder
    states_value = encoder_model.predict(input_seq)

    # Crear la secuencia inicial del decoder
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = output_tokenizer.word_index["<start>"]

    # Almacenar la respuesta generada
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Obtener el índice del token predicho
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_output_word_index.get(sampled_token_index, "")

        decoded_sentence += " " + sampled_word

        # Condición de parada: token de fin o longitud máxima alcanzada
        if sampled_word == "<end>" or len(decoded_sentence.split()) > max_output_length:
            stop_condition = True

        # Actualizar la secuencia objetivo
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Actualizar los estados
        states_value = [h, c]

    return decoded_sentence
tf.keras.models.save_model(model, 'model.h5')
tf.saved_model.save(model, 'sample_data/tf_model')
# ===================
# Probar el chatbot
# ===================
def chat_with_bot(input_text):
    input_seq = pad_sequences(input_tokenizer.texts_to_sequences([input_text]), maxlen=max_input_length, padding="post")
    response = decode_sequence(input_seq)
    return response.replace("<start>", "").replace("<end>", "").strip()



# Ejemplo de interacción
print(chat_with_bot("Hola"))
print("----")
print(chat_with_bot("¿Cómo estás?"))

print("----")
print(chat_with_bot("Cuéntame un chiste"))

import json

# Exportar input_tokenizer a JSON
input_tokenizer_json = input_tokenizer.to_json()
with open("input_tokenizer.json", "w") as f:
    f.write(input_tokenizer_json)

# Exportar output_tokenizer a JSON
output_tokenizer_json = output_tokenizer.to_json()
with open("output_tokenizer.json", "w") as f:
    f.write(output_tokenizer_json)

print("Tokenizadores exportados como input_tokenizer.json y output_tokenizer.json")
```

* El propósito principal es crear un chatbot que pueda procesar texto en español, generando respuestas coherentes mediante aprendizaje supervisado. La implementación incluye desde el preprocesamiento de datos hasta la inferencia para interactuar con el usuario.
* Para los componentes principales se cuenta con los siguientes
    * Datos de Entrada y Salida: Estos utilizan las listas de textos para poder simular preguntas y respuestas, en donde las respuestas están marcadas mediante etiquetas para indicar de esta forma el inicio y el fin para cada secuencia.
    * Preprocesamiento: Los textos para esto se convierten en secuencias numéricas tokenizandose, esto se aplicatambien para que todas las secuencias tengan una misma longitud y de esta forma tener una salida la cual se codifica en one-hot encoding para ser compatibles con el modelo..
    * Modelo Seq2Seq:
        * Encoder: Este toma las secuencias de entrada, procesandolas y generando de esta forma dos estados, los cuales se encargan de desglozar y resumir la información de entrada.
        * Decoder: Utiliza los estados del encoder para un contexto inicial, generando las palabras de salida.
    * Entrenamiento: Este modelo se entrena con una función de perdida categórica, la cual es adecuada para la predicción de palabras de un vocabulario. 
    * Inferencia: En este el encoder se encarga de generar estados a partir de una entrada mientras que el decoder genra palabras las cuales están basadas en dichos estados y las predicciones previas.
    * Interacción con el Chatbot: Permite al usuario interactuar con el chatbox, tomando una entrada, analizandola y prediciendo la respuesta, devuelve un texto generado.

* Para las funcionalidades del código se tiene lo siguiente:
    * Entrenamiento del modelo: Este utiliza pares de preguntas y respuestas para enseñar al modelo a predecir una salida basada en una entrada, podiendo permitir al usuario a introducir un texto y poder de esta forma obtener respuestas generadas por el mismo.
* Resultados Esperados: Se puede decir que se pueden obtener respuestas simples y coherentes basadas en los datos de entrenamiento proporcionados, en lo que es la capacidad de aprender patrones, si lleva su buena cantidad de preguntas pero poco a poco aprende patrones en lo que es la relación de pregunta-respuesta.

###  GenCode.ipynb

```py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================
# Configuración de parámetros
# ============================
latent_dim = 256  # Reducido para menor consumo de memoria
num_samples = 10000  # Número máximo de muestras
max_input_length = 20  # Reducido para mejorar eficiencia
max_output_length = 200

# Leer el archivo TSV en español con pandas
file_path_es_algo = "massive_dataset_es.tsv"
data_es_algo = pd.read_csv(file_path_es_algo, sep="\t")

# Leer el archivo TSV en inglés con pandas
file_path_en_algo = "massive_dataset_en.tsv"
data_en_algo = pd.read_csv(file_path_en_algo, sep="\t")

# Extraer columnas "Question" y "Answer"
input_texts_es_algo = data_es_algo["Question"].tolist()
output_texts_es_algo = ["<start> " + str(answer_es) + " <end>" for answer_es in data_es_algo["Answer"].tolist()]

input_texts_en_algo = data_en_algo["Question"].tolist()
output_texts_en_algo = ["<start> " + str(answer_en) + " <end>" for answer_en in data_en_algo["Answer"].tolist()]

# Unificación de datasets
input_texts = input_texts_es_algo + input_texts_en_algo
output_texts = output_texts_es_algo + output_texts_en_algo

# ========================
# Preprocesamiento de datos
# ========================
# Tokenización de las secuencias
input_tokenizer = Tokenizer()
output_tokenizer = Tokenizer(filters="")

input_tokenizer.fit_on_texts(input_texts)
output_tokenizer.fit_on_texts(output_texts)

input_sequences = input_tokenizer.texts_to_sequences(input_texts)
output_sequences = output_tokenizer.texts_to_sequences(output_texts)

# Agregar padding para las secuencias
encoder_input_data = pad_sequences(input_sequences, maxlen=max_input_length, padding="post")
decoder_input_data = pad_sequences([seq[:-1] for seq in output_sequences], maxlen=max_output_length, padding="post")
decoder_target_data = pad_sequences([seq[1:] for seq in output_sequences], maxlen=max_output_length, padding="post")

# Dividir los datos en entrenamiento y validación
encoder_train, encoder_val, decoder_input_train, decoder_input_val, decoder_target_train, decoder_target_val = train_test_split(
    encoder_input_data, decoder_input_data, decoder_target_data, test_size=0.2, random_state=42
)

# ===================
# Construcción del modelo
# ===================
# Encoder
encoder_inputs = Input(shape=(None,), dtype="int32")
encoder_embedding = tf.keras.layers.Embedding(input_dim=len(input_tokenizer.word_index) + 1,
                                               output_dim=latent_dim,
                                               mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,), dtype="int32")
decoder_embedding = tf.keras.layers.Embedding(input_dim=len(output_tokenizer.word_index) + 1,
                                               output_dim=latent_dim,
                                               mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(len(output_tokenizer.word_index) + 1, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

# Modelo Seq2Seq
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compilar el modelo
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Resumen del modelo
model.summary()

# ========================
# Generador de datos
# ========================
def data_generator(encoder_data, decoder_input_data, decoder_target_data, batch_size):
    while True:
        for i in range(0, len(encoder_data), batch_size):
            yield ([encoder_data[i:i + batch_size], decoder_input_data[i:i + batch_size]],
                   decoder_target_data[i:i + batch_size])

# Generadores para entrenamiento y validación
batch_size = 16
train_generator = data_generator(encoder_train, decoder_input_train, decoder_target_train, batch_size)
val_generator = data_generator(encoder_val, decoder_input_val, decoder_target_val, batch_size)

def data_generator(encoder_data, decoder_input_data, decoder_target_data, batch_size):
    def generator():
        for i in range(0, len(encoder_data), batch_size):
            # Convert NumPy arrays to tf.int32 tensors
            encoder_input = tf.cast(encoder_data[i:i + batch_size], dtype=tf.int32)
            decoder_input = tf.cast(decoder_input_data[i:i + batch_size], dtype=tf.int32)
            decoder_target = tf.cast(decoder_target_data[i:i + batch_size], dtype=tf.int32)

            # Yield data in the expected structure (tuple of tuples)
            yield ((encoder_input, decoder_input), decoder_target)
    return generator

# Crear datasets usando tf.data.Dataset.from_generator
batch_size = 16

train_dataset = tf.data.Dataset.from_generator(
    data_generator(encoder_train, decoder_input_train, decoder_target_train, batch_size),
    output_signature=(
        (
            tf.TensorSpec(shape=(None, max_input_length), dtype=tf.int32),
            tf.TensorSpec(shape=(None, max_output_length), dtype=tf.int32),
        ),
        tf.TensorSpec(shape=(None, max_output_length), dtype=tf.int32)
    )
).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    data_generator(encoder_val, decoder_input_val, decoder_target_val, batch_size),
    output_signature=(
        (
            tf.TensorSpec(shape=(None, max_input_length), dtype=tf.int32),
            tf.TensorSpec(shape=(None, max_output_length), dtype=tf.int32),
        ),
        tf.TensorSpec(shape=(None, max_output_length), dtype=tf.int32)
    )
).prefetch(tf.data.AUTOTUNE)

# Entrenar el modelo usando los datasets
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=[
        EarlyStopping(patience=5, monitor="val_loss"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
    ]
)

# ===================
# Modelos para inferencia
# ===================
# Encoder para inferencia
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder para inferencia
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_lstm_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_lstm_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states,
)

# ===================
# Función para decodificar secuencias
# ===================
reverse_input_word_index = dict((i, word) for word, i in input_tokenizer.word_index.items())
reverse_output_word_index = dict((i, word) for word, i in output_tokenizer.word_index.items())

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = output_tokenizer.word_index["<start>"]

    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_output_word_index.get(sampled_token_index, "")

        decoded_sentence += " " + sampled_word

        if sampled_word == "<end>" or len(decoded_sentence.split()) > max_output_length:
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return decoded_sentence

# Guardar el modelo
model.save("seq2seq_model.h5")
tf.saved_model.save(model, 'tf_model')

# Exportar tokenizadores a JSON
with open("input_tokenizer.json", "w") as f:
    f.write(input_tokenizer.to_json())

with open("output_tokenizer.json", "w") as f:
    f.write(output_tokenizer.to_json())

print("Model and tokenizers exported successfully!")
```

* El propósito principal es crear un chatbot que pueda procesar texto en español, generando respuestas coherentes respecto el código por medio del aprendizaje supervisado. La implementación incluye desde el preprocesamiento de datos hasta la inferencia para interactuar con el usuario.
* Para los componentes principales se cuenta con los siguientes
    * Datos de Entrada y Salida: Estos utilizan las listas de textos (en este caso el código para poder analizar cual algoritmo es cual) para poder simular preguntas y respuestas, en donde las respuestas están marcadas mediante etiquetas para indicar de esta forma el inicio y el fin para cada secuencia.
    * Preprocesamiento: Los textos para esto se convierten en secuencias numéricas tokenizandose, esto se aplicatambien para que todas las secuencias tengan una misma longitud y de esta forma tener una salida la cual se codifica en one-hot encoding para ser compatibles con el modelo..
    * Modelo Seq2Seq:
        * Encoder: Este toma las secuencias de entrada, procesandolas y generando de esta forma dos estados, los cuales se encargan de desglozar y resumir la información de entrada.
        * Decoder: Utiliza los estados del encoder para un contexto inicial, generando las palabras de salida.
    * Entrenamiento: Este modelo se entrena con una función de perdida categórica, la cual es adecuada para la predicción de palabras para código. 
    * Inferencia: En este el encoder se encarga de generar estados a partir de una entrada mientras que el decoder genra palabras las cuales están basadas en dichos estados y las predicciones previas.
    * Interacción con el Chatbot: Permite al usuario interactuar con el chatbox, tomando una entrada, analizandola y prediciendo la respuesta, devolviendo el código según sea lo que se le pida.

* Para las funcionalidades del código se tiene lo siguiente:
    * Entrenamiento del modelo: Este utiliza pares de preguntas y respuestas para enseñar al modelo a predecir una salida basada en una entrada, podiendo permitir al usuario a introducir un texto y poder de esta forma obtener respuestas generadas por el mismo.
* Resultados Esperados: Se puede decir que se pueden obtener respuestas en código correspondientes para los tipos de algoritmos que se piden, todo ello basado en los datos de entrenamiento que le fueron proporcionados y en la capacidad de aprender patrones.


### DATASET
Dado que ahora es necesario que el chatbot entienda tanto el idioma en inglés como el español, se han implementado 2 datasets para tener un mejor orden, uno por cada idioma, además, se a modificado levemente su estructura con la intención de ser mas intuitivas a la hora de leerlos.

#### EJEMPLIFICACIÓN DEL DATASTE EN ESPAÑOL [[dataset_es.tsv](../../Proyecto/Fase1/modelo/dataset/dataset_es.tsv)]
```
Question	Answer
¿Eres mayor que yo?	En realidad, no tengo edad. Los bots no cumplimos años.
Eres una bebé	En realidad, no tengo edad. Los bots no cumplimos años.
Dime cuál es tu edad	En realidad, no tengo edad. Los bots no cumplimos años.
....
Me siento súper baldado	He oído que no hay nada como una buena siesta.
Me siento súper cansada	He oído que no hay nada como una buena siesta.
Me siento súper hastiado	He oído que no hay nada como una buena siesta.
```
#### EJEMPLIFICACIÓN DEL DATASTE EN INGLES [[dataset_en.tsv](../../Proyecto/Fase1/modelo/dataset/dataset_en.tsv)]
```
Question	Answer
Are you older than me?	Actually, I'm not old. Bots don't have a birthday.
You're a baby	Actually, I'm not old. Bots don't have a birthday.
Tell me your age	Actually, I'm not old. Bots don't have a birthday.
....
I feel super bald	I've heard there's nothing like a good nap.
I feel super tired	I've heard there's nothing like a good nap.
I feel super jaded	I've heard there's nothing like a good nap.

```

Este dataset fue modificado y adaptado tomando como base el siguiente:
```
https://qnamakerstore.blob.core.windows.net/qnamakerdata/editorial/spanish/qna_chitchat_friendly.tsv
```

#### EJEMPLIFICACIÓN DEL DATASET PARA CÓDIGO [[dataset.tsv](../../Proyecto/Fase1/modelo/datasets_coding/massive_dataset_es.tsv)]

Question	Answer
¿Me puedes dar el metodo bubble sort?	<<Generación del código de bubble sort>>
¿Me puedes dar el metodo quick sort?	<<Generación del código de bubble sort>>


### Código Aplicación Tkinter

#### app.py
```py
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from model.ChatBotModel import ChatBotModel

class ChatApp:
    def __init__(self, root, chatbot_model, chatbotcode_model):
        self.root = root
        self.root.title("IA CHAT")
        self.root.geometry("500x700")
        self.root.configure(bg="#F5F5F5")

        self.chatbot_model = chatbot_model
        self.chatbotcode_model = chatbotcode_model
        self.use_code_model = tk.BooleanVar(value=False)

        # Preguntas comunes
        self.common_questions = {
            "es": [
                "¿Eres mayor que yo?",
                "Eres una bebé",
                "Dime cuál es tu edad",
                "Di si eres un chiquitín",
                "Di si eres una pequeña",
                "¿Cómo me puedes ayudar?",
                "¿Qué tipo de ayuda puedes ofrecerme?",
                "¿Qué apoyo puedes darme?",
                "Dime cómo puedes ayudarme",
                "¿Cómo puedes ser útil para mí?",
            ],
            "en": [
                "Are you older than me?",
                "You're a baby",
                "Tell me your age",
                "Say if you are a little boy",
                "Say if you're a little girl",
                "How can you help me?",
                "What kind of help can you offer me?",
                "What support can you give me?",
                "Tell me how you can help me",
                "How can you be useful to me?",
            ],
        }
        self.selected_language = tk.StringVar(value="es")

        # Model selector
        self.create_model_selector()

        # Language selector
        self.create_language_selector()

        # Chat frame
        self.chat_frame_container = tk.Frame(self.root, bg="#F5F5F5")
        self.chat_frame_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))

        self.chat_frame = tk.Canvas(self.chat_frame_container, bg="#F5F5F5", highlightthickness=0)
        self.chat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = ttk.Scrollbar(self.chat_frame_container, orient="vertical", command=self.chat_frame.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_frame.configure(yscrollcommand=self.scrollbar.set)

        self.messages_frame = tk.Frame(self.chat_frame, bg="#F5F5F5")
        self.chat_frame.create_window((0, 0), window=self.messages_frame, anchor="nw")

        self.messages_frame.bind("<Configure>", lambda e: self.chat_frame.configure(scrollregion=self.chat_frame.bbox("all")))

        # Entry area
        self.entry_frame = tk.Frame(self.root, bg="#F5F5F5")
        self.entry_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

        self.entry_field = tk.Entry(self.entry_frame, font=("Helvetica", 12), bd=0, relief=tk.FLAT, bg="#E8E8E8", fg="#333")
        self.entry_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10), pady=5, ipady=5)
        self.entry_field.bind("<Return>", self.send_message)

        self.send_button = tk.Button(
            self.entry_frame,
            text="Enviar",
            font=("Helvetica", 12),
            bg="#4CAF50",
            fg="#FFF",
            bd=0,
            relief=tk.FLAT,
            command=self.send_message
        )
        self.send_button.pack(side=tk.RIGHT, pady=5, ipady=5, ipadx=10)

        # Dropdown para preguntas comunes
        self.create_question_dropdown()

        # Initial messages
        self.initial_messages = [
            {
                "sender": "Chat-Ia",
                "text": "Hola, soy IaChatbot. Estoy aquí para ayudarte con cualquier pregunta que tengas.",
                "type": "received",
            },
            {
                "sender": "Chat-Ia",
                "text": "He sido entrenado para responder en español e inglés. ¡Prueba a preguntarme algo!",
                "type": "received",
            }
        ]
        self.load_initial_messages()

    def create_model_selector(self):
        model_frame = tk.Frame(self.root, bg="#F5F5F5")
        model_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        tk.Label(
            model_frame,
            text="Selecciona el modelo:",
            bg="#F5F5F5",
            font=("Helvetica", 12),
        ).pack(side=tk.LEFT, padx=(0, 10))

        self.selected_model = tk.StringVar(value="chatbot")
        self.model_dropdown = ttk.Combobox(
            model_frame,
            textvariable=self.selected_model,
            values=["chatbot", "chatcode"],
            state="readonly",
            font=("Helvetica", 12),
        )
        self.model_dropdown.pack(side=tk.LEFT)
        self.model_dropdown.bind("<<ComboboxSelected>>", self.update_selected_model)

    def update_selected_model(self, event=None):
        if self.selected_model.get() == "chatcode":
            self.use_code_model.set(True)
        else:
            self.use_code_model.set(False)
        self.update_question_dropdown()  # Actualizar las preguntas según el modelo seleccionado

    def create_language_selector(self):
        language_frame = tk.Frame(self.root, bg="#F5F5F5")
        language_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        tk.Label(
            language_frame,
            text="Selecciona un idioma:",
            bg="#F5F5F5",
            font=("Helvetica", 12),
        ).pack(side=tk.LEFT, padx=(0, 10))

        self.language_dropdown = ttk.Combobox(
            language_frame,
            textvariable=self.selected_language,
            values=["es", "en"],
            state="readonly",
            font=("Helvetica", 12),
        )
        self.language_dropdown.pack(side=tk.LEFT)
        self.language_dropdown.bind("<<ComboboxSelected>>", self.update_question_dropdown)

    def create_question_dropdown(self):
        question_frame = tk.Frame(self.root, bg="#F5F5F5")
        question_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

        tk.Label(
            question_frame,
            text="Preguntas comunes:",
            bg="#F5F5F5",
            font=("Helvetica", 12, "bold"),
        ).pack(anchor="w", pady=(0, 5))

        self.dropdown_var = tk.StringVar(value="Selecciona una pregunta")
        self.question_dropdown = ttk.Combobox(
            question_frame,
            textvariable=self.dropdown_var,
            state="readonly",
            font=("Helvetica", 10),
        )
        self.question_dropdown.pack(fill=tk.X)
        self.question_dropdown.bind("<<ComboboxSelected>>", self.insert_dropdown_question)

        self.update_question_dropdown()

    def update_question_dropdown(self, event=None):
        language = self.selected_language.get()
        if self.selected_model.get() == "chatcode":
            # Preguntas específicas para `chatcode` (Python y JavaScript)
            questions = {
                "es": [
                    # Python
                    "¿Cómo imprimo en pantalla con Python?",
                    "Genera el código para declarar variables de tipo entero en Python.",
                    "Genera el código para declarar constantes de tipo entero en Python.",
                    "Dame el código para generar un array unidimensional de cadenas en Python.",
                    "Dame el código para generar un array bidimensional de cadenas en Python.",
                    "¿Cómo puedo escribir un bucle Do en Python?",
                    "Crea una declaración if usando mayor que en Python.",
                    "Dame un ejemplo de if else if else en Python.",
                    "Dame un ejemplo de switch en Python.",
                    "¿Cómo implementas una función en Python?",
                    "¿Puedes proporcionar el algoritmo de Bubble Sort en Python?",
                    "Extrae los ifs del algoritmo de Bubble Sort.",
                    # JavaScript
                    "¿Cómo imprimo en consola con JavaScript?",
                    "Genera el código para declarar variables de tipo entero en JavaScript.",
                    "Genera el código para declarar constantes de tipo entero en JavaScript.",
                    "Dame el código para generar un array unidimensional de cadenas en JavaScript.",
                    "Dame el código para generar un array bidimensional de cadenas en JavaScript.",
                    "¿Cómo puedo escribir un bucle Do en JavaScript?",
                    "Crea una declaración if usando mayor que en JavaScript.",
                    "Dame un ejemplo de if else if else en JavaScript.",
                    "Dame un ejemplo de switch en JavaScript.",
                    "¿Cómo implementas una función en JavaScript?",
                    "¿Puedes proporcionar el algoritmo de Bubble Sort en JavaScript?",
                    "Extrae los ifs del algoritmo de Bubble Sort en JavaScript.",
                ],
                "en": [
                    # Python
                    "How do I print on screen with Python?",
                    "Generate the code to declare variables of type integer in Python.",
                    "Generate the code to declare constants of type integer in Python.",
                    "Give me the code to generate a one-dimensional array of strings in Python.",
                    "Give me the code to generate a two-dimensional array of strings in Python.",
                    "How can I write a Do loop in Python?",
                    "Create the if statement using greater than in Python.",
                    "Give me an example of if else if else in Python.",
                    "Give me an example of switch in Python.",
                    "How do you implement a function in Python?",
                    "Can you provide the Bubble Sort algorithm in Python?",
                    "Extract the ifs of the Bubble Sort algorithm.",
                    # JavaScript
                    "How do I print to the console in JavaScript?",
                    "Generate the code to declare variables of type integer in JavaScript.",
                    "Generate the code to declare constants of type integer in JavaScript.",
                    "Give me the code to generate a one-dimensional array of strings in JavaScript.",
                    "Give me the code to generate a two-dimensional array of strings in JavaScript.",
                    "How can I write a Do loop in JavaScript?",
                    "Create the if statement using greater than in JavaScript.",
                    "Give me an example of if else if else in JavaScript.",
                    "Give me an example of switch in JavaScript.",
                    "How do you implement a function in JavaScript?",
                    "Can you provide the Bubble Sort algorithm in JavaScript?",
                    "Extract the ifs of the Bubble Sort algorithm in JavaScript.",
                ],
            }.get(language, [])
        else:
            # Preguntas generales para `chatbot`
            questions = self.common_questions.get(language, [])

        self.question_dropdown["values"] = questions
        self.question_dropdown.set("Selecciona una pregunta")

    def insert_dropdown_question(self, event=None):
        selected_question = self.dropdown_var.get()
        if selected_question and selected_question != "Selecciona una pregunta":
            self.entry_field.delete(0, tk.END)
            self.entry_field.insert(0, selected_question)

    def load_initial_messages(self):
        for message in self.initial_messages:
            self.display_message(message["sender"], message["text"], align="left")

    def send_message(self, event=None):
        user_message = self.entry_field.get().strip()
        if user_message:
            self.display_message("You", user_message, align="right")
            self.entry_field.delete(0, tk.END)

            selected_model = self.chatbotcode_model if self.use_code_model.get() else self.chatbot_model
            try:
                bot_response = selected_model.chat(user_message)
                if not bot_response:
                    bot_response = "Lo siento, no entiendo. ¿Podrías reformular tu pregunta?"
            except Exception as e:
                print(f"Error getting response: {e}")
                bot_response = "Lo siento, ha ocurrido un error. Por favor, inténtalo de nuevo."

            self.display_message("Chat-Ia", bot_response, align="left")

    def copy_to_clipboard(self, text):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()

    def display_message(self, sender, message, align="left"):
        frame = tk.Frame(self.messages_frame, bg="#F5F5F5")
        frame.pack(fill=tk.X, pady=8, padx=(10 if align == "left" else 50, 10 if align == "right" else 50), anchor="w" if align == "left" else "e")

        bg_color = "#4CAF50" if align == "right" else "#E8E8E8"
        fg_color = "#FFF" if align == "right" else "#333"

        message_frame = tk.Frame(frame, bg=bg_color)
        message_frame.pack(fill=tk.NONE, side=tk.LEFT if align == "left" else tk.RIGHT, padx=5)

        label = tk.Label(
            message_frame,
            text=message,
            bg=bg_color,
            fg=fg_color,
            wraplength=350,
            justify="left" if align == "left" else "right",
            font=("Helvetica", 12),
            padx=10,
            pady=5
        )
        label.pack(fill=tk.X)

        if align == "left":
            copy_button = tk.Button(
                message_frame,
                text="Copiar",
                font=("Helvetica", 8),
                bg="#D3D3D3",
                fg="#000",
                bd=0,
                command=lambda: self.copy_to_clipboard(message)
            )
            copy_button.pack(side=tk.RIGHT, padx=5, pady=5)

        timestamp = datetime.now().strftime("%H:%M")
        time_label = tk.Label(
            frame,
            text=timestamp,
            bg="#F5F5F5",
            fg="#888",
            font=("Helvetica", 8),
            anchor="e" if align == "right" else "w"
        )
        time_label.pack(anchor="w" if align == "left" else "e")

        self.root.after(100, lambda: self.chat_frame.yview_moveto(1.0))

if __name__ == "__main__":
    try:
        chatbot_model = ChatBotModel(
            'model/chatModel/tf_model', 
            'model/chatModel/input_tokenizer.json', 
            'model/chatModel/output_tokenizer.json', 
            max_output_length=200
        )

        chatbotcode_model = ChatBotModel(
            'model/chatCodeModel/tf_model', 
            'model/chatCodeModel/input_tokenizer.json', 
            'model/chatCodeModel/output_tokenizer.json', 
            max_output_length=200
        )

        root = tk.Tk()
        app = ChatApp(root, chatbot_model, chatbotcode_model)
        root.mainloop()
    except Exception as e:
        print(f"Error initializing application: {e}")
```

En el siguiente código se muestran los métodos de la clase `ChatApp` para lo que es el funcionamiento de la aplicación, todo ello con lo que son importaciones referentes a los modelos para su implementación posterior, se definen en *__init__* todos los componentes visuales y como irán

#### ChatBotModel.py
```py
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json


class ChatBotModel:
    def __init__(self, model_path, input_tokenizer_path, output_tokenizer_path, latent_dim=256, max_input_length=20, max_output_length=20):
        """
        Inicializa el chatbot cargando el modelo, los tokenizadores y configurando parámetros.
        """
        self.latent_dim = latent_dim
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        # Cargar el modelo guardado
        try:
            self.model = tf.saved_model.load(model_path)
        except Exception as e:
            raise RuntimeError(f"Error al cargar el modelo desde {model_path}: {e}")

        # Cargar los tokenizadores desde JSON
        try:
            with open(input_tokenizer_path, "r") as f:
                input_tokenizer_json = f.read()
            self.input_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(input_tokenizer_json)

            with open(output_tokenizer_path, "r") as f:
                output_tokenizer_json = f.read()
            self.output_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(output_tokenizer_json)
        except Exception as e:
            raise RuntimeError(f"Error al cargar los tokenizadores: {e}")

        # Reconstruir índices inversos
        self.reverse_input_word_index = {i: word for word, i in self.input_tokenizer.word_index.items()}
        self.reverse_output_word_index = {i: word for word, i in self.output_tokenizer.word_index.items()}

        # Obtener tokens especiales
        self.start_token = self.output_tokenizer.word_index.get("<start>")
        self.end_token = self.output_tokenizer.word_index.get("<end>")
        if self.start_token is None or self.end_token is None:
            raise ValueError("Los tokens '<start>' y '<end>' deben estar presentes en el output_tokenizer.")

        # Obtener la firma de inferencia del modelo
        if "serving_default" not in self.model.signatures:
            raise ValueError("La firma 'serving_default' no está presente en el modelo cargado.")
        self.serving_default = self.model.signatures["serving_default"]

    def text_to_sequences(self, text):
        """
        Convierte texto en una secuencia de índices utilizando el tokenizador.
        """
        sequence = []
        word_index = self.input_tokenizer.word_index


        # Preprocesar texto: convertir a minúsculas, manejar puntuación
        tokens = text.lower().replace(",", " ,").replace(".", " .").replace("?", " ?").split()
        for word in tokens:
            if word in word_index:
                sequence.append(word_index[word])
        return sequence

    def pad_sequences(self, sequence, max_len):
        """
        Aplica padding a la secuencia para alcanzar la longitud máxima.
        """
        padded = [0] * max_len
        for i, value in enumerate(sequence[:max_len]):
            padded[i] = value
        return padded

    def decode_output(self, indices):
        """
        Decodifica una lista de índices en texto utilizando el tokenizador de salida.
        """
        response = []
        for index in indices:
            if index in self.reverse_output_word_index:
                response.append(self.reverse_output_word_index[index])
        return " ".join(response)

    def chat(self, input_text):
        """
        Genera una respuesta para el texto de entrada.
        """
        # Preparar entrada del encoder
        input_sequence = self.text_to_sequences(input_text)
        padded_sequence = self.pad_sequences(input_sequence, self.max_input_length)
        encoder_input_tensor = tf.constant([padded_sequence], dtype=tf.int32)

        decoder_input = [self.start_token]
        response = []
        stop_condition = False

        while not stop_condition and len(decoder_input) <= self.max_output_length:
            # Preparar entrada del decoder
            padded_decoder_input = self.pad_sequences(decoder_input, self.max_output_length)
            decoder_input_tensor = tf.constant([padded_decoder_input], dtype=tf.int32)

            # Ejecutar inferencia
            try:
                output_tokens = self.model.signatures["serving_default"](
                    inputs=encoder_input_tensor, inputs_1=decoder_input_tensor
                )["output_0"].numpy()
            except Exception as e:
                raise RuntimeError(f"Error durante la inferencia: {e}")

            # Obtener el siguiente token
            next_token = np.argmax(output_tokens[0, len(decoder_input) - 1, :])
            if next_token == self.end_token or len(response) >= self.max_output_length:
                stop_condition = True
            else:
                response.append(next_token)
                decoder_input.append(next_token)

        # Decodificar respuesta
        return self.decode_output(response).replace("<start>", "").replace("<end>", "").replace("\\n","\n").replace("\\t","\t").strip()
```

Aqui está otra clase `ChatBotModel`, la cual se encarga de importar los modelos para el chat (a parte de app.py) y se encarga de obtener las respuestas de los modelos que se utilicen.