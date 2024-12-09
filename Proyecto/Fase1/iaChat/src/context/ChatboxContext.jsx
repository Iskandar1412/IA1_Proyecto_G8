import React, { createContext, useContext, useState } from 'react';
import { useModelChatBot } from '../hooks/useModelChatBot';

const ChatbotContext = createContext();

const initialMessages = [
  {
    id: 0,
    sender: 'C',
    text: 'Hola, soy IaChatbot, ¿en qué puedo ayudarte? 🤖',
    type: 'received',
  },
];

export const ChatbotProvider = ({ children }) => {
  const {generarRespuesta, isLoading:isLoadingModel} = useModelChatBot();

  const [messages, setMessages] = useState(initialMessages);
  const [inputValue, setInputValue] = useState('');

  const [error, setError] = useState(null);

  const sendMessage = async (text) => {
    const newMessage = {
      id: messages.length + 1,
      sender: 'A',
      text,
      type: 'sent',
    };

    setMessages((prevMessages) => [...prevMessages, newMessage]);
    setInputValue('');


    try {
      handleReceivedMessage('Escribiendo...');
      const chatbotResponse = await generarRespuesta(text);
      // eliminar el mensaje de "escribiendo..."
      setMessages((prevMessages) => prevMessages.slice(0, -1));
      handleReceivedMessage(chatbotResponse);
    } catch (err) {
      console.error('Error al manejar la entrada del chatbot:', err);
      setError('Hubo un problema al procesar tu mensaje.');
    } finally {



    }
  };

  const handleReceivedMessage = (text) => {
    const newMessage = {
      id: messages.length + 1,
      sender: 'C',
      text,
      type: 'received',
    };
    setMessages((prevMessages) => [...prevMessages, newMessage]);
  };

  return (
    <ChatbotContext.Provider
      value={{
        messages,
        inputValue,
        setInputValue,
        sendMessage,
        error,
        isLoadingModel
      }}
    >
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
