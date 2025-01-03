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
      const loadedModel = await tf.loadGraphModel("./tfjs_model/model.json");
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
        .toLowerCase() // Convertir a minúsculas
        .replace(/([.,!?¿¡])/g, " $1 ") // Separar signos de puntuación
        .split(/\s+/); // Dividir en palabras eliminando espacios extra
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
