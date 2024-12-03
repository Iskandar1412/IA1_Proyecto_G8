import { createBrowserRouter } from "react-router-dom";
import { Chat } from "../pages/Chat";
import { Layout } from "../containers/chat/Layout";
import { ChatbotProvider } from "../context/ChatboxContext";

const trainingData = [
  {
    input: "Crea un bucle for en Python",
    output: "for i in range(10):\n    print(i)",
  },
  {
    input: "Define una función en JavaScript",
    output: 'function greet() {\n    console.log("Hello, World!");\n}',
  },
];

const router = createBrowserRouter([
  {
    path: "/",
    element: (
 
        <Layout>
          <Chat />
        </Layout>
  
    ),
  },
]);

export default router;
