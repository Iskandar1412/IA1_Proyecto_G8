# Manual Técnico

## Introducción

En el desarrollo de la primera fase del proyecto de Inteligencia Artificial 1, se implementó una solución utilizando React y Vite como tecnologías principales para construir una interfaz gráfica interactiva que permita la comunicación con un modelo de inteligencia artificial. Este modelo, inicialmente desarrollado en Python, fue adaptado y trasladado a JavaScript para integrarse con la aplicación web.

La funcionalidad principal del sistema consiste en recibir entradas de texto en español, procesarlas a través de un modelo previamente entrenado y proporcionar respuestas coherentes en tiempo real. Para este propósito, se emplearon herramientas de conversión y serialización para traducir el modelo desarrollado en Python a un formato compatible con JavaScript, garantizando una integración fluida en el entorno del navegador.

El enfoque del proyecto incluyó la creación de una estructura modular en React, lo que facilitó la gestión de componentes y el manejo del estado, mientras que Vite proporcionó un entorno de desarrollo optimizado y de alto rendimiento. Adicionalmente, se diseñaron flujos de datos que conectan la interfaz gráfica con el modelo, permitiendo una interacción eficiente y de fácil uso.

Este proyecto demostró la capacidad de implementar un modelo de inteligencia artificial funcional en un entorno web moderno, asegurando un alto nivel de rendimiento y accesibilidad para los usuarios.

## Requisitos del Sistema

> **Systema Operativo:** Windows, Ubuntu 22.04 o superior, arch linux
> **CPU:** Intel Pentium D o AMD Athlon 64 (K8) 2.6GHz o superior
> **RAM:** 4GB
> **Lenguajes Utilizados:** React
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

latent_dim = 256 
num_samples = 10000
max_input_length = 20
max_output_length = 20 

input_texts = [
    "Hola",
    "¿Cómo estás?",
    "¿Qué haces?",
    "Adiós",
    "¿Cuál es tu nombre?",
    "¿Qué día es hoy?",
    "Cuéntame un chiste",
    "¿Dónde estás?",
    "¿Qué te gusta hacer?",
    "Gracias",
    "Buenos días",
    "Buenas noches",
    "¿Cuál es tu color favorito?",
    "¿Puedes ayudarme?",
    "¿Cómo funciona esto?",
    "Dime algo interesante",
    "¿Qué hora es?",
    "Recomiéndame un libro",
    "¿Cuál es tu comida favorita?",
]

output_texts = [
    "<start> Hola <end>",
    "<start> Estoy bien, gracias <end>",
    "<start> Nada en particular <end>",
    "<start> Hasta luego <end>",
    "<start> Mi nombre es ChatBot <end>",
    "<start> Hoy es un gran día <end>",
    "<start> ¿Qué hace una abeja en el gimnasio? ¡Zum-ba! <end>",
    "<start> Estoy aquí, contigo <end>",
    "<start> Me gusta aprender cosas nuevas <end>",
    "<start> De nada <end>",
    "<start> Buenos días, ¿cómo estás? <end>",
    "<start> Que tengas buenas noches <end>",
    "<start> Mi color favorito es el azul <end>",
    "<start> Claro, dime en qué necesitas ayuda <end>",
    "<start> Esto funciona con inteligencia artificial <end>",
    "<start> El sol es una estrella increíblemente grande <end>",
    "<start> No llevo reloj, pero es un buen momento <end>",
    "<start> Te recomiendo 'El principito' <end>",
    "<start> Me encanta la pizza <end>",
]

input_tokenizer = Tokenizer()
output_tokenizer = Tokenizer(filters="")  

input_tokenizer.fit_on_texts(input_texts)
output_tokenizer.fit_on_texts(output_texts)

input_sequences = input_tokenizer.texts_to_sequences(input_texts)
output_sequences = output_tokenizer.texts_to_sequences(output_texts)
encoder_input_data = pad_sequences(input_sequences, maxlen=max_input_length, padding="post")
decoder_input_data = pad_sequences([seq[:-1] for seq in output_sequences], maxlen=max_output_length, padding="post")
decoder_target_data = pad_sequences([seq[1:] for seq in output_sequences], maxlen=max_output_length, padding="post")
num_decoder_tokens = len(output_tokenizer.word_index) + 1
decoder_target_data = tf.keras.utils.to_categorical(decoder_target_data, num_decoder_tokens)

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
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=64,
    epochs=100,
    validation_split=0.2,
)

encoder_model = Model(encoder_inputs, encoder_states)
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
tf.keras.models.save_model(model, 'model.h5')
tf.saved_model.save(model, 'sample_data/tf_model')
def chat_with_bot(input_text):
    input_seq = pad_sequences(input_tokenizer.texts_to_sequences([input_text]), maxlen=max_input_length, padding="post")
    response = decode_sequence(input_seq)
    return response.replace("<start>", "").replace("<end>", "").strip()


print(chat_with_bot("Hola"))
print("----")
print(chat_with_bot("¿Cómo estás?"))

print("----")
print(chat_with_bot("Cuéntame un chiste"))

import json
input_tokenizer_json = input_tokenizer.to_json()
with open("input_tokenizer.json", "w") as f:
    f.write(input_tokenizer_json)

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

### Modelo Chatbot (modelChatbot.js)

```js
import * as tf from '@tensorflow/tfjs';
import * as use from '@tensorflow-models/universal-sentence-encoder';
import data from './data.json';

const { entradas, respuestas } = data;

let model = null;

export async function cargarModelo() {
    if (!model) {
        model = await use.load();
        console.log('Modelo cargado');
    }
    return model;
}

export async function reconocerInput(userInput) {
    if (!model) throw new Error('El modelo no está cargado.');
    const userInputEmb = await model.embed([userInput]);
    let punteo = -1;
    let entradaReconocida = null;

    for (const [entrada, examples] of Object.entries(entradas)) {
        const examplesEmb = await model.embed(examples);
        const scores = await tf.matMul(userInputEmb, examplesEmb, false, true).data();
        const maxExampleScore = Math.max(...scores);
        if (maxExampleScore > punteo) {
            punteo = maxExampleScore;
            entradaReconocida = entrada;
        }
    }

    return entradaReconocida;
}

export async function generarRespuesta(userInput) {
    const entrada = await reconocerInput(userInput);
    if (entrada && respuestas[entrada]) {
        return respuestas[entrada];
    } else {
        return 'Lo siento, no logro comprenderte, ¿Puedes volver a repetirlo?';
    }
}
```

* Para este modelo se convierten frases en vectores numéricos para permitir la similitud entre las frases de manera más eficiente, de forma que el modelo se carga una única vez y se puede reutilizar para procesar multiples entradas, y mejorando el rendimiento del mismo.
* Cuando el usuario realiza una entrada (userInput), el sistema convierte dicha entrada y la representa utilizando el modelo USE. Posteriormente compara dicha representacón con un conjunto de ejemplos almacenados en el archivo JSON (data.json) el cual contiene las entradas predefinidas agrupadas por categorías, etc.
* Para cada categoría, calcula una métrica de similitud entre el texto ingresado y los ejemplos asociados; permitiendo de cierta forma el determinar qué categoría de entrada coincide mejor con la intención del usuario.
* Una vez el modelo identifica la categoria o intención del usuario escrito en el texto, el sistema se encarga de buscar una respuesta asociada a dicha categoria; para el caso que no encuentre una categoria adecuada, devolverá un mensaje predeterminado en el cual le dice al usuario que no comprende la entrada realizada anteriormente.

### Comando para exportación modelo (Python a JS)

```bash
!tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --control_flow_v2=true  sample_data/tf_model/ sample_data/tfjs_model2
```

## Concluciones

* El modelo Seq2Seq basado en LSTM demostró ser efectivo para la generación de respuestas en lenguaje natural, aprendiendo directamente de los datos etiquetados; demostrando su capacidad para poder generalizar y adaptarse a nuevos contextos, convirtiendolo en una solución robusta para dominios complejos.
* El chatbot basado en Universal Sentence Encoder es más rápido en la implementación y rápidez de manejar tareas de clasifcación de texto en tiempo real, haciendo de mencion sus limintaciones en ejemplos predefinidos, siendo una solución eficiente y escalable para sistemas que requieren interacciones rápidas y controladas.
* Los dos modelos cumplen el propósito que requieren para lo que es la interacción con usuarios, resultando el modelo Seq2Sec como un modelo ideal para escenarios dinámicos y generativos y el segundo modelo para interacciones predefinidas y controladas.
