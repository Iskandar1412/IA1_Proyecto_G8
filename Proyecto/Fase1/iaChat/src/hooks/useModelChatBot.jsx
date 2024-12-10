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
      const loadedModel = await tf.loadLayersModel("/tfjs_model/model.json");
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
    
    // Parsear tokenizer.config.word_index solo una vez al inicio
    const wordIndex = JSON.parse(tokenizer.config.word_index);
    
    text.split(" ").forEach((word) => {
      // Verificar si la palabra existe en el índice
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
    // Parsear outputTokenizer.config.word_index solo una vez al inicio
    const wordIndex = JSON.parse(outputTokenizer.config.word_index);
    
    indices.forEach((index) => {
      // Buscar la palabra correspondiente al índice
      for (const [word, idx] of Object.entries(wordIndex)) {
        if (idx == index) {
          console.log(idx + "-" + word)
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

    const maxInputLength = 20; // Longitud máxima definida en el modelo
    const maxOutputLength = 20; // Longitud máxima definida en el modelo

    // Verificar y asignar los tokens <start> y <end>
    let startToken = JSON.parse(outputTokenizer.config.word_index)["<start>"];
    let endToken = JSON.parse(outputTokenizer.config.word_index)["<end>"];

    if (startToken === undefined || endToken === undefined) {
      console.warn("Tokens <start> o <end> no encontrados. Configurando manualmente.");
      // Configurar manualmente si no están presentes
      startToken = 1; // Generalmente es 1
      endToken = 2;   // Generalmente es 2
    }

    // Preparar entrada del encoder
    const inputSequence = textsToSequences(inputTokenizer, inputText);
    const paddedSequence = padSequences(inputSequence, maxInputLength);

    const encoderInputTensor = tf.tensor2d([paddedSequence], [1, maxInputLength]);

    let decoderInputSequence = [startToken];
    let response = "";
    let stopCondition = false;

    while (!stopCondition && decoderInputSequence.length <= maxOutputLength) {
      const paddedDecoderInput = padSequences(decoderInputSequence, maxOutputLength);
      const decoderInputTensor = tf.tensor2d([paddedDecoderInput], [1, maxOutputLength]);

      const outputTokens = model.predict([encoderInputTensor, decoderInputTensor]);
      const outputIndices = outputTokens.argMax(-1).dataSync();

      const nextToken = outputIndices[decoderInputSequence.length - 1];

      if (nextToken === endToken || response.split(" ").length >= maxOutputLength) {
        stopCondition = true;
      } else {
        response += decodeOutput([nextToken]) + " ";
        decoderInputSequence.push(nextToken);
      }
    }
    
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
