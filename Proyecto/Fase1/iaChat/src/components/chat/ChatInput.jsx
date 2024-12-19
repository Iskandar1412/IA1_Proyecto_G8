import React, { useState } from "react";
import { useChatbotContext } from "../../context/ChatboxContext";

export const ChatInput = () => {
  const { sendMessage, setInputValue, inputValue } = useChatbotContext();
  const [language, setLanguage] = useState("es"); // Estado para idioma seleccionado

  // Preguntas comunes en español e inglés
  const commonQuestions = {
    es: [
      "¿Eres mayor que yo?",
      "Eres una bebé",
      "Dime cuál es tu edad",
      "Di si eres un chiquitín",
      "Di si eres una pequeña",
      "¿Cómo me puedes ayudar?",
      "¿Qué tipo de ayuda puedes ofrecerme?",
      "¿Qué apoyo puedes darme?",
      "Dime cómo puedes ayudarme",
      "¿Cómo puedes ser útil para mí?",
    ],
    en: [
      "Are you older than me?",
      "You're a baby",
      "Tell me your age",
      "Say if you are a little boy",
      "Say if you're a little girl",
      "How can you help me?",
      "What kind of help can you offer me?",
      "What support can you give me?",
      "Tell me how you can help me",
      "How can you be useful to me?",
    ],
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputValue.trim()) {
      sendMessage(inputValue);
      setInputValue("");
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="flex flex-row items-center h-16 rounded-xl bg-white w-full px-4"
    >
      {/* Selector de idioma */}
      <div className="mr-4">
        <select
          className="border rounded-lg p-2"
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
        >
          <option value="es">Español</option>
          <option value="en">English</option>
        </select>
      </div>

      {/* Selector de preguntas comunes */}
      <div className="mr-4">
        <select
          className="border rounded-lg p-2"
          onChange={(e) => setInputValue(e.target.value)}
          value=""
        >
          <option value="" disabled>
            Preguntas comunes
          </option>
          {commonQuestions[language].map((question, index) => (
            <option key={index} value={question}>
              {question}
            </option>
          ))}
        </select>
      </div>

      {/* Campo de texto */}
      <div className="flex-grow ml-4">
        <div className="relative w-full">
          <input
            type="text"
            className="flex w-full border rounded-xl focus:outline-none focus:border-indigo-300 pl-4 h-10"
            value={inputValue}
            placeholder="Type your message..."
            onChange={(e) => setInputValue(e.target.value)}
          />
        </div>
      </div>

      {/* Botón de enviar */}
      <div className="ml-4">
        <button
          type="submit"
          className="flex items-center justify-center bg-indigo-500 hover:bg-indigo-600 rounded-xl text-white px-4 py-1 flex-shrink-0"
        >
          <span>Send</span>
          <span className="ml-2">
            <svg
              className="w-4 h-4 transform rotate-45 -mt-px"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
              ></path>
            </svg>
          </span>
        </button>
      </div>
    </form>
  );
};
