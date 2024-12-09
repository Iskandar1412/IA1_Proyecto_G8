import React from "react";
import { ChatMessage } from "./ChatMessage";
import { useChatbotContext } from "../../context/ChatboxContext";

export const ChatMessages = () => {
  const { messages, isLoadingModel } = useChatbotContext();

  return (
    <div className="grid grid-cols-12 gap-y-2">
      {isLoadingModel ? (
        <div className="col-span-12 flex flex-col items-center justify-center text-gray-500">
          <div className="loader ease-linear rounded-full border-4 border-t-4 border-gray-200 h-12 w-12 mb-4"></div>
          Cargando modelo... Por favor espera.
        </div>
      ) : (
        messages.map((message) => (
          <ChatMessage key={message.id} {...message} />
        ))
      )}
    </div>
  );
};
