import React, { createContext, useContext, useState, useEffect } from 'react';



const ChatbotContext = createContext();
const initialMessages = [

  { id: 1, sender: "C", text: "Hola, son IaChatbot, en que puedo ayudarte?🤖", type: "received" },
];



export const ChatbotProvider = ({  children }) => {

  const [messages, setMessages] = useState(initialMessages);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isTyping, setIsTyping] = useState(false);

  const sendMessage = (text) => {
    const newMessage = {
      id: messages.length + 1,
      sender: 'A',
      text,
      type: 'sent',
    };
    setMessages((prevMessages) => [...prevMessages, newMessage]);
    setInputValue('');
    setIsTyping(true);
    setTimeout(() => {
      setIsTyping(false);
      setMessages((prevMessages) =>
        prevMessages.map((msg) =>
          msg.id === newMessage.id ? { ...msg, seen: true } : msg
        )
      );

      handleReceivedMessage('Hola, son IaChatbot, en que puedo ayudarte?🤖');
    }, 1000);
  
  }

  const handleReceivedMessage = (text) => {
    const newMessage = {
      id: messages.length + 1,
      sender: 'C',
      text,
      type: 'received',
    };
    setMessages((prevMessages) => [...prevMessages, newMessage]);
  }

 return (
    <ChatbotContext.Provider
      value={{
        messages,
        setMessages,
        inputValue,
        setInputValue,
        loading,
        setLoading,
        error,
        setError,
        isTyping,
        setIsTyping,
        sendMessage
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
