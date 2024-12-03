import React from "react";
import { ChatInput } from "./ChatInput";
import { ChatMessages } from "./ChatMessages";
import { ChatHeader } from "./ChatHeader";

export const ChatBox = () => {
  return (
    <div class="flex h-screen antialiased text-gray-800 ">
      <div class="flex flex-row h-full w-full overflow-x-hidden">
        <div class="flex flex-col flex-auto h-full p-6">
          <div class="flex flex-col flex-auto flex-shrink-0 rounded-2xl bg-gray-100 h-full p-4">
            <ChatHeader />
            <div class="flex flex-col h-full overflow-x-auto mb-4 overflow-scroll">
              <div class="flex flex-col h-full">
                <ChatMessages />
              </div>
            </div>
            <ChatInput />
          </div>
        </div>
      </div>
    </div>
  );
};
