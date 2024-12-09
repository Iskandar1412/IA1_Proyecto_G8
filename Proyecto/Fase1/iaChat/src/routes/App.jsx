
import { RouterProvider } from 'react-router-dom'
import router from './router'

import { Chat } from "../pages/Chat";
import { Layout } from "../containers/chat/Layout";
import { ChatbotProvider } from "../context/ChatboxContext";

export const App = () => {
  // return (
  //   <RouterProvider router={router} />
  // )

  return(
    <ChatbotProvider>
    <Layout>
      <Chat />
    </Layout>
  </ChatbotProvider>
  )
}