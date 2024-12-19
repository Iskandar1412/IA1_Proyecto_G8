import * as tf from '@tensorflow/tfjs';
import * as use from '@tensorflow-models/universal-sentence-encoder';
import data from './data.json'; // Importar las entradas y respuestas desde el JSON

const { entradas, respuestas } = data;

let model = null;

// Cargar el modelo
export async function cargarModelo() {
    if (!model) {
        model = await use.load();
        console.log('Modelo cargado');
    }
    return model;
}

// Reconocer entrada del usuario
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

// Generar respuesta
export async function generarRespuesta(userInput) {
    const entrada = await reconocerInput(userInput);
    if (entrada && respuestas[entrada]) {
        return respuestas[entrada];
    } else {
        return 'Lo siento, no logro comprenderte, ¿Puedes volver a repetirlo?';
    }
}