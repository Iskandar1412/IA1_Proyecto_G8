# Manual Técnico

## Introducción

En el desarrollo de la primera fase del proyecto de Inteligencia Artificial 1, se implementó una solución utilizando React y Vite como tecnologías principales para construir una interfaz gráfica interactiva que permita la comunicación con un modelo de inteligencia artificial. Este modelo, inicialmente desarrollado en Python, fue adaptado y trasladado a JavaScript para integrarse con la aplicación web.

La funcionalidad principal del sistema consiste en recibir entradas de texto en español, procesarlas a través de un modelo previamente entrenado y proporcionar respuestas coherentes en tiempo real. Para este propósito, se emplearon herramientas de conversión y serialización para traducir el modelo desarrollado en Python a un formato compatible con JavaScript, garantizando una integración fluida en el entorno del navegador.

El enfoque del proyecto incluyó la creación de una estructura modular en React, lo que facilitó la gestión de componentes y el manejo del estado, mientras que Vite proporcionó un entorno de desarrollo optimizado y de alto rendimiento. Adicionalmente, se diseñaron flujos de datos que conectan la interfaz gráfica con el modelo, permitiendo una interacción eficiente y de fácil uso.

Este proyecto demostró la capacidad de implementar un modelo de inteligencia artificial funcional en un entorno web moderno, asegurando un alto nivel de rendimiento y accesibilidad para los usuarios.

## Requisitos Mínimos del Sistema

> **Systema Operativo:** Windows 7 o superior, Ubuntu 22.04 o superior, arch linux
> **CPU:** Intel Pentium D o AMD Athlon 64 (K8) 2.6GHz o superior
> **RAM:** 8GB
> **Framework Frontend:** React + Vite (npm version 10.9.2)
> **Lenguaje Modelo:** JavaScript
> **Lenguaje Modelo:** Python (version 3.12.7)
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

### Comando para exportación modelo (Python a JS)
Dado que el modelo fue entrenado en python, es necesario realizar la exportación a js por cuestiones de restricciones del proyecto, este comando se encarga de la conversión del modelo para exportarlo a JavaScript.
```bash
!tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --control_flow_v2=true  sample_data/tf_model/ sample_data/tfjs_model2
```

### Uso Del Modelo (useModelChatBot.jsx)
Este código corresponde a un custom hook de React que se utiliza para interactuar con el modelo de TensorFlow.js para crear un chatbot. A continuación, te explico cada sección del código de manera detallada.
```js
import { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import inputTokenizer from "./input_tokenizer.json";
import outputTokenizer from "./output_tokenizer.json";

export const useModelChatBot = () => {
  const [model, setModel] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  // Función para cargar el modelo
  const cargarModelo = async () => {
    try {
      const loadedModel = await tf.loadGraphModel("/tfjs_model/model.json");
      setModel(loadedModel);
      setIsLoading(false);
      console.log("Modelo cargado correctamente");
    } catch (error) {
      console.error("Error al cargar el modelo:", error);
      setIsLoading(false);
    }
  };

  // Tokenización: convertir texto a secuencia de números con preprocesamiento mejorado
  const textoASecuencias = (tokenizer, text) => {
    const sequence = [];
    const wordIndex = JSON.parse(tokenizer.config.word_index);

    // Preprocesar texto: manejar signos especiales y convertir a minúsculas
    const tokenizeText = (text) => {
      return text
        .toLowerCase()
        .replace(/([.,!?¿¡])/g, " $1 ")
        .split(/\s+/);
    };

    tokenizeText(text).forEach((word) => {
      if (wordIndex[word] !== undefined) {
        sequence.push(wordIndex[word]);
      }
    });

    return sequence;
  };

  // Padding de secuencias
  const padSequences = (sequence, maxLen) => {
    const padded = Array(maxLen).fill(0);
    sequence.slice(0, maxLen).forEach((value, idx) => (padded[idx] = value));
    return padded;
  };

  // Decodificación: convertir predicciones a texto
  const decodeOutput = (indices) => {
    let response = "";
    const wordIndex = JSON.parse(outputTokenizer.config.word_index);
    
    // Invertir el mapa de índices para buscar palabras
    const indexToWord = Object.fromEntries(
      Object.entries(wordIndex).map(([word, idx]) => [idx, word])
    );

    indices.forEach((index) => {
      if (indexToWord[index]) {
        response += indexToWord[index] + " ";
      }
    });

    return response.trim();
  };

  // Función para generar respuesta
  const generarRespuesta = async (inputText) => {
    if (!model) {
      console.error("El modelo no está cargado aún");
      return "El modelo no está listo";
    }

    const maxInputLength = 20;
    const maxOutputLength = 20;

    let startToken = JSON.parse(outputTokenizer.config.word_index)["<start>"];
    let endToken = JSON.parse(outputTokenizer.config.word_index)["<end>"];
    if (startToken === undefined || endToken === undefined) {
      console.warn("Tokens <start> o <end> no encontrados. Configurando manualmente.");
      startToken = 1;
      endToken = 2;
    }

    // Preparar entrada del encoder
    const inputSequence = textoASecuencias(inputTokenizer, inputText.toLowerCase());
    const paddedSequence = padSequences(inputSequence, maxInputLength);
    const encoderInputTensor = tf.tensor2d([paddedSequence], [1, maxInputLength], "int32");

    let decodificarSecuenciaInput = [startToken];
    let response = "";
    let stopCondition = false;

    while (!stopCondition && decodificarSecuenciaInput.length <= maxOutputLength) {
      const paddedDecoderInput = padSequences(decodificarSecuenciaInput, maxOutputLength);
      const decoderInputTensor = tf.tensor2d([paddedDecoderInput], [1, maxOutputLength], "int32");
      const outputTokens = await model.executeAsync([encoderInputTensor, decoderInputTensor]);

      const outputTensor = Array.isArray(outputTokens) ? outputTokens[0] : outputTokens;
      const outputIndices = outputTensor.argMax(-1).dataSync();
      const nextToken = outputIndices[decodificarSecuenciaInput.length - 1];

      if (nextToken === endToken || response.split(" ").length >= maxOutputLength) {
        stopCondition = true;
      } else {
        response += decodeOutput([nextToken]) + " ";
        decodificarSecuenciaInput.push(nextToken);
      }

      tf.dispose(outputTokens);
    }

    return response.replace("<start>", "").replace("<end>", "").trim();
  };

  useEffect(() => {
    cargarModelo();
  }, []);

  return {
    isLoading,
    generarRespuesta,
  };
};

```
* **cargarModelo**: Es una función que intenta cargar un modelo de TensorFlow desde el path (/tfjs_model/model.json), además de esto, la función permite indicar al código si el modelo fue cargado o no para su posterior manipulación.
* **textoASecuencias** : Convierte el texto en una secuencia de números utilizando un tokenizador.
    * **tokenizeText**: Se encarga de convertir el texto a minúsculas, reemplazar signos de puntuación por espacios y dividir el texto en palabras, para que luego convertir cada palabra en un número usando el índice de la palabra en el tokenizador.
* **padSequences**: Se encarga de asegurar que las secuencias tengan una longitud fija, si la secuencia es más corta, se completa con ceros. Si es más larga, se recorta.
* **decodeOutput**: Convierte los índices de las predicciones del modelo de vuelta a palabras utilizando el word_index del tokenizador de salida.
  * Se invierte el **wordIndex** para obtener un mapeo de índices a palabras, los índices de las predicciones se convierten en palabras y se ensamblan en una cadena de texto.
* **generarRespuesta**: Es la función principal para generar la respuesta del chatbot. Acepta un texto de entrada y genera una respuesta.
  *  Primero, verifica si el modelo está cargado.
  *  Luego, procesa el texto de entrada con la tokenización y el padding, y prepara las secuencias para el modelo.
  *  **decodificarSecuenciaInput**: Comienza con el token de inicio, y el modelo genera palabras sucesivas hasta que alcanza el token de fin.
  *  Para cada predicción, se decodifica el índice a palabra, y se agrega a la respuesta.
  *  El proceso se repite hasta que se genera una respuesta completa.
  
### LINK DE COLAB DEL MODELO
```
https://colab.research.google.com/drive/1THxV3Qy_JrS1Vrf0ufBHD1JMMlAaA64g?usp=sharing
```
