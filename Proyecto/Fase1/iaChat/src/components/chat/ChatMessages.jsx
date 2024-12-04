import React, { useState } from "react";
import { ChatMessage } from "./ChatMessage";
import { useChatbotContext } from "../../context/ChatboxContext";

export const ChatMessages = () => {
  const {messages} = useChatbotContext();

  return (
    <div className="grid grid-cols-12 gap-y-2">
      {messages.map((message) => (
        <ChatMessage key={message.id} {...message} />
      ))}
    </div>
  );
};
