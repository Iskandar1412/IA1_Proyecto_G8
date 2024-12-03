import React from "react";
import { ChatMessage } from "./ChatMessage";

export const ChatMessages = () => {
  const messages = [
    { id: 1, sender: "C", text: "Hey How are you today?", type: "received" },
    {
      id: 2,
      sender: "C",
      text: "Lorem ipsum dolor sit amet, consectetur adipisicing elit.",
      type: "received",
    },
    { id: 3, sender: "A", text: "I'm ok what about you?", type: "sent" },
    {
      id: 4,
      sender: "A",
      text: "Lorem ipsum dolor sit, amet consectetur adipisicing.",
      type: "sent",
    },
    {
      id: 5,
      sender: "C",
      text: "Lorem ipsum dolor sit amet!",
      type: "received",
    },
    {
      id: 6,
      sender: "A",
      text: "Lorem ipsum dolor sit, amet consectetur adipisicing.",
      type: "sent",
      seen: true,
    },
    {
      id: 7,
      sender: "C",
      text: "Lorem ipsum dolor sit amet consectetur adipisicing elit.",
      type: "received",
    },
    {
      id: 7,
      sender: "C",
      text: "Lorem ipsum dolor sit amet consectetur adipisicing elit.",
      type: "received",
    },
    {
      id: 7,
      sender: "C",
      text: "Lorem ipsum dolor sit amet consectetur adipisicing elit.",
      type: "received",
    },
    {
      id: 7,
      sender: "C",
      text: "Lorem ipsum dolor sit amet consectetur adipisicing elit.",
      type: "received",
    },
  ];

  return (
    <div className="grid grid-cols-12 gap-y-2">
      {messages.map((message) => (
        <ChatMessage key={message.id} {...message} />
      ))}
    </div>
  );
};
