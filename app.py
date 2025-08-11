import pathlib
import os
from dotenv import load_dotenv
import gradio as gr
import google.genai as genai
from google.genai import types

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

# Create GenAI client
client = genai.Client(api_key=api_key)

# Upload Indian Constitution PDF once
file_path = pathlib.Path('resources/20240716890312078.pdf')  # fixed typo if needed
if not file_path.exists():
    raise FileNotFoundError(f"PDF not found at {file_path}")

sample_file = client.files.upload(file=file_path)
print(f"📄 PDF uploaded with ID: {sample_file.name}")

# Initial system instruction
system_prompt = (
            "You are a friendly, experienced constitutional lawyer. "
            "You are speaking to a client who has little legal knowledge. "
            "Answer me straight and don't include unnecessary legal jargon."
            "Your job is to explain answers in clear, simple language, while telling me what I can do to get help."
            "ask relevent questions to understand the user's needs better."
        )
  

# Chat history
chat_history = [{"role": "system", "content": system_prompt}]

# Streaming function for Gradio
def chat_with_lawyer(message, history):
    # Append user message
    chat_history.append({"role": "user", "content": message})

    # Prepare full conversation
    conversation = [sample_file] + [m["content"] for m in chat_history]

    # Stream reply
    reply_text = ""
    for chunk in client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=conversation
    ):
        if chunk.text:
            reply_text += chunk.text
            yield reply_text  # Send partial text to Gradio

    # Save reply to history
    chat_history.append({"role": "assistant", "content": reply_text})


# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## ⚖️ Indian Constitution Lawyer\nAsk legal questions based on the uploaded PDF.")
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(label="Your Question", placeholder="Type your legal question here...")
    clear = gr.Button("Clear Chat")

    def respond(message, chat_ui_history):
        stream = chat_with_lawyer(message, chat_ui_history)
        full_reply = ""
        for partial in stream:
            full_reply = partial
            yield chat_ui_history + [(message, partial)], ""  # clear textbox

    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: (None, ""), None, [chatbot, msg], queue=False)

demo.queue()
demo.launch(share=True)


