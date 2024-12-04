import React from "react";

export const ChatHeader = () => {
  return (
    <div className="flex items-center justify-between p-4 bg-indigo-800 text-white rounded-t-xl mb-4">
      <div className="flex items-center space-x-3">
        <div className="w-10 h-10 bg-white rounded-full flex items-center justify-center">
        🤖
        </div>
        <div>
          <h1 className="text-lg font-semibold">IA Chat Bot</h1>
          <p className="text-sm text-gray-200">Tu asistente inteligente</p>
        </div>
      </div>

    </div>
  );
};
