import { createBrowserRouter } from "react-router-dom";
import { Chat } from "../pages/Chat";
import { Layout } from "../containers/Layout";

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
