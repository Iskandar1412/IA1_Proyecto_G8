import React, { createContext, useContext, useState, useEffect } from 'react';
import { createAndTrainModel, generateCode } from '../model/modelChatbot';


const ChatbotContext = createContext();


export const ChatbotProvider = ({ trainingData, children }) => {
  const [isTraining, setIsTraining] = useState(false);

  useEffect(() => {
    const trainModel = async () => {
      setIsTraining(true);
      await createAndTrainModel(trainingData);
      setIsTraining(false);
    };
    trainModel();
  }, [trainingData]);

  return (
    <ChatbotContext.Provider value={{ isTraining, generateCode }}>
      {children}
    </ChatbotContext.Provider>
  );
};


export const useChatbotContext = () => {
  const context = useContext(ChatbotContext);
  if (!context) {
    throw new Error('useChatbotContext debe usarse dentro de un ChatbotProvider');
  }
  return context;
};
