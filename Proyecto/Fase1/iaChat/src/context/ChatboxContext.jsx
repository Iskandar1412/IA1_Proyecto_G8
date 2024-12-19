import React, { createContext, useContext, useState } from 'react';
import { useModelChatBot } from '../hooks/useModelChatBot';

const ChatbotContext = createContext();
const initialMessages = [
  {
    id: 4440,
    sender: 'C',
    text: 'Hola, soy IaChatbot. Estoy aquí para ayudarte con cualquier pregunta que tengas. 🤖',
    type: 'received',
  },
  {
    id: 4441,
    sender: 'C',
    text: 'Puedes preguntarme cosas como:\n- ¿Eres mayor que yo?\n- ¿Cómo me puedes ayudar?\n- Dime cuál es tu edad.',
    type: 'received',
  },
  {
    id: 44442,
    sender: 'C',
    text: 'También puedo responder preguntas sobre diversos temas o simplemente charlar. ¡Escríbeme algo para empezar!',
    type: 'received',
  },
  {
    id:4424223,
    sender: 'C',
    text: 'Ah, y también sé algo de inglés, aunque sigo aprendiendo. Puedes intentar preguntarme algo en ese idioma. 😊',
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
