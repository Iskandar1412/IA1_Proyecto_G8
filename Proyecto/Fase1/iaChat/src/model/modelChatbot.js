import * as tf from '@tensorflow/tfjs';

let model;
let vocab = [];


const tokenize = (text) => text.split(' ').map((word) => vocab.indexOf(word) + 1 || 0);


export async function createAndTrainModel(trainingData) {

  vocab = [...new Set(trainingData.flatMap((d) => d.input.split(' ')))];


  const xs = tf.tensor(trainingData.map((d) => tokenize(d.input)));
  const ys = tf.tensor(trainingData.map((d) => tokenize(d.output)));

  model = tf.sequential();
  model.add(tf.layers.embedding({ inputDim: vocab.length + 1, outputDim: 50 }));
  model.add(tf.layers.lstm({ units: 256, returnSequences: true }));
  model.add(tf.layers.dense({ units: vocab.length, activation: 'softmax' }));

  model.compile({
    optimizer: 'adam',
    loss: 'sparseCategoricalCrossentropy',
  });


  await model.fit(xs, ys, { epochs: 10, batchSize: 2 });
}


export async function generateCode(prompt) {
  if (!model || !vocab.length) throw new Error('El modelo no está entrenado.');
  const inputTensor = tf.tensor([tokenize(prompt)]);
  const prediction = model.predict(inputTensor);
  const outputTokens = prediction.argMax(-1).dataSync();
  return outputTokens.map((token) => vocab[token]).join(' ');
}
