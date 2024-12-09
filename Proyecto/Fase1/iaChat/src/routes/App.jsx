
import { Chat } from "../pages/Chat";
import { Layout } from "../containers/chat/Layout";
import { ChatbotProvider } from "../context/ChatboxContext";

export const App = () => {
  return (
    // <RouterProvider router={router} />
    <ChatbotProvider>
        <Layout>
          <Chat />
        </Layout>
    </ChatbotProvider>
  )
}