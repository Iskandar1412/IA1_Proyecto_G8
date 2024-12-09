import React from "react";
import { useChatbotContext } from "../../context/ChatboxContext";

export const ChatInput = () => {
  const { sendMessage, setInputValue, inputValue } = useChatbotContext();
  
  const handleSubmit = (e) => {
    e.preventDefault(); // Evita que el formulario recargue la página
    if (inputValue.trim()) {
      sendMessage(inputValue); // Envía el mensaje usando el contexto
      setInputValue(""); // Limpia el campo de entrada
    }
  };

  return (
    <form
      onSubmit={handleSubmit} // Maneja el evento de envío
      className="flex flex-row items-center h-16 rounded-xl bg-white w-full px-4"
    >
      <div className="flex-grow ml-4">
        <div className="relative w-full">
          <input
            type="text"
            className="flex w-full border rounded-xl focus:outline-none focus:border-indigo-300 pl-4 h-10"
            value={inputValue}
            placeholder="Type your message..."
            onChange={(e) => setInputValue(e.target.value)} // Actualiza el estado del input
          />
        </div>
      </div>
      <div className="ml-4">
        <button
          type="submit" // Cambiado a "submit" para integrarse con el formulario
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
