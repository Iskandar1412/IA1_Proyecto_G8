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
        return self.decode_output(response).replace("<start>", "").replace("<end>", "").strip()

