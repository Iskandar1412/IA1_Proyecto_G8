import tkinter as tk
from tkinter import ttk
from datetime import datetime
from model.ChatBotModel import ChatBotModel

class ChatApp:
    def __init__(self, root, chatbot_model):
        self.root = root
        self.root.title("IA CHAT 🤖")
        self.root.geometry("500x600")
        self.root.configure(bg="#2b2b2b")
        
        self.chatbot_model = chatbot_model
   
        self.chat_frame = tk.Frame(self.root, bg="#3c3f41", padx=10, pady=10)
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))

        self.text_widget = tk.Text(self.chat_frame, wrap=tk.WORD, state=tk.DISABLED,
                                   bg="#2b2b2b", fg="#ffffff", font=("Helvetica", 12),
                                   relief=tk.FLAT, highlightthickness=0, padx=10, pady=10, spacing1=2)
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = ttk.Scrollbar(self.chat_frame, command=self.text_widget.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_widget["yscrollcommand"] = self.scrollbar.set

        # Entry area
        self.entry_frame = tk.Frame(self.root, bg="#2b2b2b")
        self.entry_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

        self.entry_field = ttk.Entry(self.entry_frame, font=("Helvetica", 12))
        self.entry_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.entry_field.bind("<Return>", self.send_message)

        self.send_button = ttk.Button(self.entry_frame, text="Enviar", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT)

        # Styling
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TButton", font=("Helvetica", 12), background="#4CAF50", foreground="#ffffff")
        self.style.map("TButton", background=[("active", "#45a049")])

        # Load initial messages
        self.initial_messages = [
            {
                "sender": "Chat-Ia 🤖",
                "text": "Hola, soy IaChatbot. Estoy aquí para ayudarte con cualquier pregunta que tengas. 🤖",
                "type": "received",
            },
            {
                "sender": "Chat-Ia 🤖",
                "text": "He sido entrenado para responder en español e inglés. ¡Prueba a preguntarme algo! 😊",
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
            
            # Get response from the chatbot model
            try:
                bot_response = self.chatbot_model.chat(user_message)
                if not bot_response:
                    bot_response = "Lo siento, no entiendo. ¿Podrías reformular tu pregunta?"
            except Exception as e:
                print(f"Error getting response: {e}")
                bot_response = "Lo siento, ha ocurrido un error. Por favor, inténtalo de nuevo."
            
            self.display_message("Chat-Ia 🤖", bot_response)

    def display_message(self, sender, message, align="left"):
        timestamp = datetime.now().strftime("%H:%M")
        self.text_widget.config(state=tk.NORMAL)
        if align == "right":
            self.text_widget.tag_configure("user", justify="right", foreground="#4CAF50")
            self.text_widget.insert(tk.END, f"{timestamp} {sender}: {message}\n", "user")
        else:
            self.text_widget.tag_configure("bot", justify="left", foreground="#ffffff")
            self.text_widget.insert(tk.END, f"{sender}: {message} {timestamp}\n", "bot")
        self.text_widget.config(state=tk.DISABLED)
        self.text_widget.see(tk.END)


if __name__ == "__main__":
    try:
        model_path = 'model/chatCodeModel/tf_model'
        input_tokenizer_path = 'model/chatCodeModel/input_tokenizer.json'
        output_tokenizer_path = 'model/chatCodeModel/output_tokenizer.json'

        chatbot_model = ChatBotModel(model_path, input_tokenizer_path, output_tokenizer_path, max_output_length=200)



  
        root = tk.Tk()
        app = ChatApp(root, chatbot_model)
        root.mainloop()
    except Exception as e:
        print(f"Error initializing application: {e}")