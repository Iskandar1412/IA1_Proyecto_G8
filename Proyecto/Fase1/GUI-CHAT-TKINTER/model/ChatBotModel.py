import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


class ChatBotModel:
    def __init__(self, model_path, input_tokenizer_path, output_tokenizer_path, latent_dim=256, max_input_length=20, max_output_length=20):
        """
        Inicializa el chatbot cargando el modelo, los tokenizadores y configurando parámetros.
        """
        self.latent_dim = latent_dim
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
        self.model = tf.saved_model.load(model_path)
        
        with open(input_tokenizer_path, "r") as f:
            input_tokenizer_json = f.read()
        self.input_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(input_tokenizer_json)
        
        with open(output_tokenizer_path, "r") as f:
            output_tokenizer_json = f.read()
        self.output_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(output_tokenizer_json)
        
        # Reconstruir índices inversos
        self.reverse_input_word_index = {i: word for word, i in self.input_tokenizer.word_index.items()}
        self.reverse_output_word_index = {i: word for word, i in self.output_tokenizer.word_index.items()}
        
        self.serving_default = self.model.signatures["serving_default"]

    def decode_sequence(self, input_seq):
        """
        Decodifica una secuencia de entrada generando la respuesta del chatbot.
        """
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.output_tokenizer.word_index["<start>"]

        decoded_sentence = ""
        stop_condition = False

        while not stop_condition:
            # Ejecutar inferencia paso a paso
            result = self.serving_default(
                inputs=tf.constant(input_seq, dtype=tf.int32),
                inputs_1=tf.constant(target_seq, dtype=tf.int32)
            )

            output_tokens = result["output_0"].numpy()
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = self.reverse_output_word_index.get(sampled_token_index, "")

            decoded_sentence += " " + sampled_word

            if sampled_word == "<end>" or len(decoded_sentence.split()) > self.max_output_length:
                stop_condition = True

            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

        return decoded_sentence.strip()

    def chat(self, input_text):
        """
        Procesa el texto de entrada y genera la respuesta del chatbot.
        """
        input_seq = pad_sequences(
            self.input_tokenizer.texts_to_sequences([input_text]),
            maxlen=self.max_input_length,
            padding="post"
        )


        response = self.decode_sequence(input_seq)
        return response.replace("<start>", "").replace("<end>", "").strip()


