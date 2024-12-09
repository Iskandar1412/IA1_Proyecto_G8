import { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import inputTokenizer from "./input_tokenizer.json";
import outputTokenizer from "./output_tokenizer.json";

export const useModelChatBot = () => {
  const [model, setModel] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  // Función para cargar el modelo
  const loadModel = async () => {
    try {
      const loadedModel = await tf.loadGraphModel("/tfjs_model/model.json", {
      });
      console.log("Modelo cargado correctamente" ,{loadModel});
      setModel(loadedModel);
      setIsLoading(false);
      console.log("Modelo cargado correctamente");
    } catch (error) {
      console.error("Error al cargar el modelo:", error);
      setIsLoading(false);
    }
  };

  // Tokenización: convertir texto a secuencia de números
  const textsToSequences = (tokenizer, text) => {
    const sequence = [];
    text.split(" ").forEach((word) => {
      if (tokenizer.word_index[word]) {
        sequence.push(tokenizer.word_index[word]);
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
  const decodeOutput = (prediction) => {
    const indices = prediction.argMax(-1).dataSync();
    let response = "";
    indices.forEach((index) => {
      for (const [word, idx] of Object.entries(outputTokenizer.word_index)) {
        if (idx === index) {
          response += word + " ";
          break;
        }
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
    console.log("Generando respuesta para:", inputText);

    const maxInputLength = 20; // Longitud máxima definida en el modelo
    const inputSequence = textsToSequences(inputTokenizer, inputText);
    const paddedSequence = padSequences(inputSequence, maxInputLength);

    const inputTensor = tf.tensor2d([paddedSequence], [1, maxInputLength]);
    const prediction = model.predict(inputTensor);
    const response = decodeOutput(prediction);

    return response.replace("<start>", "").replace("<end>", "").trim();
  };

  // Cargar el modelo al inicio
  useEffect(() => {
    loadModel();
  }, []);

  return {
    isLoading,
    generarRespuesta,
  };
};
