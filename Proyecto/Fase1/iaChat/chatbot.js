import * as tf from '@tensorflow/tfjs';
import * as use from '@tensorflow-models/universal-sentence-encoder';

const entradas = {
    saludos: ['hola','hey','buenos días','buenas noches'],
    despedidas: ['adios','hasta la proxima','nos vemos luego'],
    agradecimientos: ['gracias','muchas gracias'],
};

const respuestas = {
    saludos: 'Hola!',
    despedidas: 'Adio!, ten un buen día',
}

let model;

async function cargarModelo() {
    model = await use.load();
    console.log('Modelo cargado');
}

async function reconocerInput(userInput) {
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

async function generarRespuesta(userInput) {
    const entrada = await reconocerInput(userInput);
    if (entrada && respuestas[entrada]) {
        return respuestas[entrada];
    } else {
        return "Lo siento, no logro comprenderte, ¿Puedes volver a repetirlo?";
    }
}

async function manejarEntrada() {
    const input = document.getElementById('user-input');
    const chatLog = document.getElementById('chat-log');

    const userInput = input.value.trim();
    if (!userInput) return;

    chatLog.innerHTML += `<div><strong>Tú:</strong> ${userInput}</div>`;
    input.value = '';

    const respuesta = await generarRespuesta(userInput);

    chatLog.innerHTML += `<div><strong>Chatbot:</strong> ${respuesta}</div>`;
    chatLog.scrollTop = chatLog.scrollHeight;
}

async function iniciarChatbot() {
    await cargarModelo();

    const sendBtn = document.getElementById('send-btn');
    sendBtn.addEventListener('click', manejarEntrada);

    const userInput = document.getElementById('user-input');
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') manejarEntrada();
    });
}

iniciarChatbot();
