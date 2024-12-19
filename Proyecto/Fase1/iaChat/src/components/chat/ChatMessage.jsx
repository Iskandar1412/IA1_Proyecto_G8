import React from "react";

export const ChatMessage = ({ sender, text, type, seen, time }) => {
  const isSent = type === "sent";

  // Formato de hora: solo horas y minutos
  const formatTime = (time) => {
    const date = time ? new Date(time) : new Date();
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div
      className={`col-start-${isSent ? "6" : "1"} col-end-${
        isSent ? "13" : "8"
      } p-3 rounded-lg`}
    >
      <div
        className={`flex items-center ${
          isSent ? "justify-start flex-row-reverse" : "flex-row"
        }`}
      >
        <div className="flex items-center justify-center h-10 w-10 rounded-full bg-indigo-500 flex-shrink-0">
          {sender}
        </div>
        <div
          className={`relative ${
            isSent ? "mr-3" : "ml-3"
          } text-sm py-2 px-4 shadow rounded-xl ${
            isSent ? "bg-indigo-100" : "bg-white"
          }`}
        >
          <div>{text}</div>
          <div className="absolute text-xs bottom-0 right-0 -mb-5 mr-2 text-gray-500">
            {formatTime(time)}
          </div>
          {seen && (
            <div className="absolute text-xs bottom-0 right-0 -mb-8 mr-2 text-gray-500">
              Seen
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
