import * as tf from "@tensorflow/tfjs";
import inputTokenizer from "./input_tokenizer.json";
import outputTokenizer from "./output_tokenizer.json";

const MODEL_PATH = "/tfjs_model/model.json";

let model = null;

// Cargar el modelo
export const loadModel = async () => {
  try {
    // Establecer el backend en 'webgl' y esperar a que esté listo
    await tf.setBackend('webgl');
    await tf.ready();

    // Cargar el modelo después de que el backend esté listo
    model = await tf.loadGraphModel(MODEL_PATH);
    console.log("Modelo cargado correctamente");
  } catch (error) {
    console.error("Error al cargar el modelo:", error);
    throw new Error("No se pudo cargar el modelo");
  }
};

// Tokenización: convertir texto a secuencia de números
export const textsToSequences = (tokenizer, text) => {
  if (!tokenizer || !tokenizer.config || !tokenizer.config.word_index) {
    console.error("Tokenizador no válido:", tokenizer);
    return [];
  }
  const sequence = [];
  text.split(" ").forEach((word) => {
    if (tokenizer.config.word_index[word]) {
      sequence.push(tokenizer.config.word_index[word]);
    }
  });
  return sequence;
};

// Padding de secuencias
export const padSequences = (sequence, maxLen) => {
  const padded = Array(maxLen).fill(0);
  sequence.slice(0, maxLen).forEach((value, idx) => (padded[idx] = value));
  return padded;
};

// Decodificación: convertir predicciones a texto
export const decodeOutput = (prediction) => {
  const indices = prediction.argMax(-1).dataSync();
  let response = "";
  for (const index of indices) {
    for (const [word, idx] of Object.entries(outputTokenizer.config.word_index)) {
      if (idx === index) {
        response += word + " ";
        break;
      }
    }
  }
  return response.trim();
};

// Generar respuesta
export const generarRespuesta = async (inputText) => {
  if (!model) {
    console.error("El modelo no está cargado aún");
    return "El modelo no está listo";
  }

  console.log("Generando respuesta para:", inputText);

  const maxInputLength = 20; // Longitud máxima definida en el modelo
  const inputSequence = textsToSequences(inputTokenizer, inputText);
  const paddedSequence = padSequences(inputSequence, maxInputLength);

  // Validar que la secuencia tiene el tamaño adecuado
  if (paddedSequence.length !== maxInputLength) {
    console.error("La secuencia de entrada no es válida:", paddedSequence);
    return "Error en el preprocesamiento de la entrada";
  }

  // Crear los tensores de entrada
  const inputTensor1 = tf.tensor2d([paddedSequence], [1, maxInputLength], "int32");
  const inputTensor2 = tf.tensor2d([paddedSequence], [1, maxInputLength], "int32");

  try {
    // Ejecutar el modelo de forma asíncrona
    const prediction = await model.executeAsync(
      {
        "inputs_1:0": inputTensor1,
        "inputs:0": inputTensor2,
      },
      ["Identity:0"]
    );
    console.log("Predicción:", prediction);

    // Decodificar la salida según sea necesario
    const response = decodeOutput(prediction);
    return response.replace("<start>", "").replace("<end>", "").trim();
  } catch (error) {
    console.error("Error durante la predicción:", error);
    return "Error al generar la respuesta.";
  } finally {
    // Liberar la memoria de los tensores
    inputTensor1.dispose();
    inputTensor2.dispose();
  }
};
