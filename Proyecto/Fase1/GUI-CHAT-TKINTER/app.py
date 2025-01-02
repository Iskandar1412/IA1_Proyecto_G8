import tkinter as tk
from tkinter import ttk
from datetime import datetime
from model.ChatBotModel import ChatBotModel

class ChatApp:
    def __init__(self, root, chatbot_model, chatbotcode_model):
        self.root = root
        self.root.title("IA CHAT")
        self.root.geometry("500x600")
        self.root.configure(bg="#F5F5F5")

        self.chatbot_model = chatbot_model
        self.chatbotcode_model = chatbotcode_model
        self.use_code_model = tk.BooleanVar(value=False)

        # Chat frame
        self.chat_frame_container = tk.Frame(self.root, bg="#F5F5F5")
        self.chat_frame_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))

        self.chat_frame = tk.Canvas(self.chat_frame_container, bg="#F5F5F5", highlightthickness=0)
        self.chat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = ttk.Scrollbar(self.chat_frame_container, orient="vertical", command=self.chat_frame.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_frame.configure(yscrollcommand=self.scrollbar.set)

        self.messages_frame = tk.Frame(self.chat_frame, bg="#F5F5F5")
        self.chat_frame.create_window((0, 0), window=self.messages_frame, anchor="nw")

        self.messages_frame.bind("<Configure>", lambda e: self.chat_frame.configure(scrollregion=self.chat_frame.bbox("all")))

        # Entry area
        self.entry_frame = tk.Frame(self.root, bg="#F5F5F5")
        self.entry_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

        self.entry_field = tk.Entry(self.entry_frame, font=("Helvetica", 12), bd=0, relief=tk.FLAT, bg="#E8E8E8", fg="#333")
        self.entry_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10), pady=5, ipady=5)
        self.entry_field.bind("<Return>", self.send_message)

        self.send_button = tk.Button(
            self.entry_frame,
            text="Enviar",
            font=("Helvetica", 12),
            bg="#4CAF50",
            fg="#FFF",
            bd=0,
            relief=tk.FLAT,
            command=self.send_message
        )
        self.send_button.pack(side=tk.RIGHT, pady=5, ipady=5, ipadx=10)

        # Initial messages
        self.initial_messages = [
            {
                "sender": "Chat-Ia",
                "text": "Hola, soy IaChatbot. Estoy aquí para ayudarte con cualquier pregunta que tengas.",
                "type": "received",
            },
            {
                "sender": "Chat-Ia",
                "text": "He sido entrenado para responder en español e inglés. ¡Prueba a preguntarme algo!",
                "type": "received",
            }
        ]
        self.load_initial_messages()

    def load_initial_messages(self):
        for message in self.initial_messages:
            self.display_message(message["sender"], message["text"], align="left")

    def send_message(self, event=None):
        user_message = self.entry_field.get().strip()
        if user_message:
            self.display_message("You", user_message, align="right")
            self.entry_field.delete(0, tk.END)

            selected_model = self.chatbotcode_model if self.use_code_model.get() else self.chatbot_model
            try:
                bot_response = selected_model.chat(user_message)
                if not bot_response:
                    bot_response = "Lo siento, no entiendo. ¿Podrías reformular tu pregunta?"
            except Exception as e:
                print(f"Error getting response: {e}")
                bot_response = "Lo siento, ha ocurrido un error. Por favor, inténtalo de nuevo."

            self.display_message("Chat-Ia", bot_response, align="left")

    def copy_to_clipboard(self, text):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()

    def display_message(self, sender, message, align="left"):
        frame = tk.Frame(self.messages_frame, bg="#F5F5F5")
        frame.pack(fill=tk.X, pady=5, padx=(10 if align == "left" else 50, 10 if align == "right" else 50), anchor="w" if align == "left" else "e")

        bg_color = "#4CAF50" if align == "right" else "#E8E8E8"
        fg_color = "#FFF" if align == "right" else "#333"

        message_frame = tk.Frame(frame, bg=bg_color)
        message_frame.pack(fill=tk.X, side=tk.LEFT, padx=5)

        label = tk.Label(
            message_frame,
            text=message,
            bg=bg_color,
            fg=fg_color,
            wraplength=350,
            justify="left" if align == "left" else "right",
            font=("Helvetica", 12),
            padx=10,
            pady=5
        )
        label.pack(fill=tk.X, anchor="w" if align == "left" else "e")

        if align == "left":
            copy_button = tk.Button(
                message_frame,
                text="Copiar",
                font=("Helvetica", 8),
                bg="#D3D3D3",
                fg="#000",
                bd=0,
                command=lambda: self.copy_to_clipboard(message)
            )
            copy_button.pack(side=tk.RIGHT, padx=5, pady=5)

        timestamp = datetime.now().strftime("%H:%M")
        time_label = tk.Label(
            frame,
            text=timestamp,
            bg="#F5F5F5",
            fg="#888",
            font=("Helvetica", 8),
            anchor="w" if align == "left" else "e"
        )
        time_label.pack(anchor="w" if align == "left" else "e")

        self.chat_frame.update_idletasks()

if __name__ == "__main__":
    try:
        chatbot_model = ChatBotModel(
            'model/chatModel/tf_model', 
            'model/chatModel/input_tokenizer.json', 
            'model/chatModel/output_tokenizer.json', 
            max_output_length=200
        )

        chatbotcode_model = ChatBotModel(
            'model/chatCodeModel/tf_model', 
            'model/chatCodeModel/input_tokenizer.json', 
            'model/chatCodeModel/output_tokenizer.json', 
            max_output_length=200
        )

        root = tk.Tk()
        app = ChatApp(root, chatbot_model, chatbotcode_model)
        root.mainloop()
    except Exception as e:
        print(f"Error initializing application: {e}")
