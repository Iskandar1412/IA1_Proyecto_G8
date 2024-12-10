import { useState, useEffect } from "react";
import { loadModel, generarRespuesta } from "../services/chatbotService";

export const useModelChatBot = () => {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const initializeModel = async () => {
      try {
        await loadModel();
        setIsLoading(false);
      } catch {
        setIsLoading(false);
      }
    };

    initializeModel();
  }, []);

  return {
    isLoading,
    generarRespuesta,
  };
};
