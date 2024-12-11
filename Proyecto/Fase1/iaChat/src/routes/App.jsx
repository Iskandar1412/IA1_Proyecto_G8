import { Chat } from "../pages/Chat";
import { Layout } from "../containers/chat/Layout";
import { ChatbotProvider } from "../context/ChatboxContext";

import { Chat } from "../pages/Chat";
import { Layout } from "../containers/chat/Layout";
import { ChatbotProvider } from "../context/ChatboxContext";

export const App = () => {
  return (
    <ChatbotProvider>
        <Layout>
          <Chat />
        </Layout>
      </ChatbotProvider>
  )
}