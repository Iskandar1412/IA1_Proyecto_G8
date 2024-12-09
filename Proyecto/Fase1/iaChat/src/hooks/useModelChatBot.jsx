import { useState, useEffect} from 'react';
import { generarRespuesta, cargarModelo } from '../model/modelChatbot';

export function useModelChatBot() {
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        async function initializeModel() {
            try {
                await cargarModelo();
                setIsLoading(false);
            } catch (error) {
                console.error('Error al cargar el modelo:', error);
                setIsLoading(false);
            }
        }
        initializeModel();
    }, []);

    return {
        isLoading,
        generarRespuesta, // Directamente expone la función para generar respuesta
    };
}
