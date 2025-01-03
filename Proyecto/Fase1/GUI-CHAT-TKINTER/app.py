import tkinter as tk
from tkinter import ttk
from datetime import datetime
from model.ChatBotModel import ChatBotModel

class ChatApp:
    def __init__(self, root, chatbot_model, chatbotcode_model):
        self.root = root
        self.root.title("IA CHAT")
        self.root.geometry("500x700")
        self.root.configure(bg="#F5F5F5")

        self.chatbot_model = chatbot_model
        self.chatbotcode_model = chatbotcode_model
        self.use_code_model = tk.BooleanVar(value=False)

        # Preguntas comunes
        self.common_questions = {
            "es": [
                "¿Eres mayor que yo?",
                "Eres una bebé",
                "Dime cuál es tu edad",
                "Di si eres un chiquitín",
                "Di si eres una pequeña",
                "¿Cómo me puedes ayudar?",
                "¿Qué tipo de ayuda puedes ofrecerme?",
                "¿Qué apoyo puedes darme?",
                "Dime cómo puedes ayudarme",
                "¿Cómo puedes ser útil para mí?",
            ],
            "en": [
                "Are you older than me?",
                "You're a baby",
                "Tell me your age",
                "Say if you are a little boy",
                "Say if you're a little girl",
                "How can you help me?",
                "What kind of help can you offer me?",
                "What support can you give me?",
                "Tell me how you can help me",
                "How can you be useful to me?",
            ],
        }
        self.selected_language = tk.StringVar(value="es")

        # Model selector
        self.create_model_selector()

        # Language selector
        self.create_language_selector()

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

        # Dropdown para preguntas comunes
        self.create_question_dropdown()

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

    def create_model_selector(self):
        model_frame = tk.Frame(self.root, bg="#F5F5F5")
        model_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        tk.Label(
            model_frame,
            text="Selecciona el modelo:",
            bg="#F5F5F5",
            font=("Helvetica", 12),
        ).pack(side=tk.LEFT, padx=(0, 10))

        self.selected_model = tk.StringVar(value="chatbot")
        self.model_dropdown = ttk.Combobox(
            model_frame,
            textvariable=self.selected_model,
            values=["chatbot", "chatcode"],
            state="readonly",
            font=("Helvetica", 12),
        )
        self.model_dropdown.pack(side=tk.LEFT)
        self.model_dropdown.bind("<<ComboboxSelected>>", self.update_selected_model)

    def update_selected_model(self, event=None):
        if self.selected_model.get() == "chatcode":
            self.use_code_model.set(True)
        else:
            self.use_code_model.set(False)
        self.update_question_dropdown()  # Actualizar las preguntas según el modelo seleccionado

    def create_language_selector(self):
        language_frame = tk.Frame(self.root, bg="#F5F5F5")
        language_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        tk.Label(
            language_frame,
            text="Selecciona un idioma:",
            bg="#F5F5F5",
            font=("Helvetica", 12),
        ).pack(side=tk.LEFT, padx=(0, 10))

        self.language_dropdown = ttk.Combobox(
            language_frame,
            textvariable=self.selected_language,
            values=["es", "en"],
            state="readonly",
            font=("Helvetica", 12),
        )
        self.language_dropdown.pack(side=tk.LEFT)
        self.language_dropdown.bind("<<ComboboxSelected>>", self.update_question_dropdown)

    def create_question_dropdown(self):
        question_frame = tk.Frame(self.root, bg="#F5F5F5")
        question_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

        tk.Label(
            question_frame,
            text="Preguntas comunes:",
            bg="#F5F5F5",
            font=("Helvetica", 12, "bold"),
        ).pack(anchor="w", pady=(0, 5))

        self.dropdown_var = tk.StringVar(value="Selecciona una pregunta")
        self.question_dropdown = ttk.Combobox(
            question_frame,
            textvariable=self.dropdown_var,
            state="readonly",
            font=("Helvetica", 10),
        )
        self.question_dropdown.pack(fill=tk.X)
        self.question_dropdown.bind("<<ComboboxSelected>>", self.insert_dropdown_question)

        self.update_question_dropdown()

    def update_question_dropdown(self, event=None):
        language = self.selected_language.get()
        if self.selected_model.get() == "chatcode":
            # Preguntas específicas para `chatcode` (Python y JavaScript)
            questions = {
                "es": [
                    # Python
                    "¿Cómo imprimo en pantalla con Python?",
                    "Genera el código para declarar variables de tipo entero en Python.",
                    "Genera el código para declarar constantes de tipo entero en Python.",
                    "Dame el código para generar un array unidimensional de cadenas en Python.",
                    "Dame el código para generar un array bidimensional de cadenas en Python.",
                    "¿Cómo puedo escribir un bucle Do en Python?",
                    "Crea una declaración if usando mayor que en Python.",
                    "Dame un ejemplo de if else if else en Python.",
                    "Dame un ejemplo de switch en Python.",
                    "¿Cómo implementas una función en Python?",
                    "¿Puedes proporcionar el algoritmo de Bubble Sort en Python?",
                    "Extrae los ifs del algoritmo de Bubble Sort.",
                    # JavaScript
                    "¿Cómo imprimo en consola con JavaScript?",
                    "Genera el código para declarar variables de tipo entero en JavaScript.",
                    "Genera el código para declarar constantes de tipo entero en JavaScript.",
                    "Dame el código para generar un array unidimensional de cadenas en JavaScript.",
                    "Dame el código para generar un array bidimensional de cadenas en JavaScript.",
                    "¿Cómo puedo escribir un bucle Do en JavaScript?",
                    "Crea una declaración if usando mayor que en JavaScript.",
                    "Dame un ejemplo de if else if else en JavaScript.",
                    "Dame un ejemplo de switch en JavaScript.",
                    "¿Cómo implementas una función en JavaScript?",
                    "¿Puedes proporcionar el algoritmo de Bubble Sort en JavaScript?",
                    "Extrae los ifs del algoritmo de Bubble Sort en JavaScript.",
                ],
                "en": [
                    # Python
                    "How do I print on screen with Python?",
                    "Generate the code to declare variables of type integer in Python.",
                    "Generate the code to declare constants of type integer in Python.",
                    "Give me the code to generate a one-dimensional array of strings in Python.",
                    "Give me the code to generate a two-dimensional array of strings in Python.",
                    "How can I write a Do loop in Python?",
                    "Create the if statement using greater than in Python.",
                    "Give me an example of if else if else in Python.",
                    "Give me an example of switch in Python.",
                    "How do you implement a function in Python?",
                    "Can you provide the Bubble Sort algorithm in Python?",
                    "Extract the ifs of the Bubble Sort algorithm.",
                    # JavaScript
                    "How do I print to the console in JavaScript?",
                    "Generate the code to declare variables of type integer in JavaScript.",
                    "Generate the code to declare constants of type integer in JavaScript.",
                    "Give me the code to generate a one-dimensional array of strings in JavaScript.",
                    "Give me the code to generate a two-dimensional array of strings in JavaScript.",
                    "How can I write a Do loop in JavaScript?",
                    "Create the if statement using greater than in JavaScript.",
                    "Give me an example of if else if else in JavaScript.",
                    "Give me an example of switch in JavaScript.",
                    "How do you implement a function in JavaScript?",
                    "Can you provide the Bubble Sort algorithm in JavaScript?",
                    "Extract the ifs of the Bubble Sort algorithm in JavaScript.",
                ],
            }.get(language, [])
        else:
            # Preguntas generales para `chatbot`
            questions = self.common_questions.get(language, [])

        self.question_dropdown["values"] = questions
        self.question_dropdown.set("Selecciona una pregunta")

    def insert_dropdown_question(self, event=None):
        selected_question = self.dropdown_var.get()
        if selected_question and selected_question != "Selecciona una pregunta":
            self.entry_field.delete(0, tk.END)
            self.entry_field.insert(0, selected_question)

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
        frame.pack(fill=tk.X, pady=8, padx=(10 if align == "left" else 50, 10 if align == "right" else 50), anchor="w" if align == "left" else "e")

        bg_color = "#4CAF50" if align == "right" else "#E8E8E8"
        fg_color = "#FFF" if align == "right" else "#333"

        message_frame = tk.Frame(frame, bg=bg_color)
        message_frame.pack(fill=tk.NONE, side=tk.LEFT if align == "left" else tk.RIGHT, padx=5)

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
        label.pack(fill=tk.X)

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
            anchor="e" if align == "right" else "w"
        )
        time_label.pack(anchor="w" if align == "left" else "e")

        self.root.after(100, lambda: self.chat_frame.yview_moveto(1.0))

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
