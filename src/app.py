import os
import sys
import shutil
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate import generate_answer
from ingest import ingest_documents

DATA_FOLDER = "data"

def upload_file(file):
    if file is None:
        return "No file uploaded."
    filename = os.path.basename(file.name)
    dest = os.path.join(DATA_FOLDER, filename)
    os.makedirs(DATA_FOLDER, exist_ok=True)
    shutil.copy(file.name, dest)
    ingest_documents()
    return f"'{filename}' uploaded and ingested successfully!"

def format_sources(sources):
    seen = set()
    result = []
    for i, source in enumerate(sources):
        if source not in seen:
            seen.add(source)
            result.append(f"[{i+1}] {source}")
    return "\n".join(result)

def query_fn(user_query: str):
    if not user_query.strip():
        return "Please enter a question first."
    answer, sources = generate_answer(user_query)
    return f"{answer}\n\nSources:\n{format_sources(sources)}"

with gr.Blocks(title="ReadMyDoc") as demo:
    gr.Markdown("# ReadMyDoc")
    gr.Markdown("Upload a document and ask questions. Answers are grounded in your content.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload Document")
            file_input = gr.File(
                label="Choose a file",
                file_types=[".pdf", ".txt", ".docx"]
            )
            upload_btn = gr.Button("Upload & Ingest", variant="secondary", size="lg")
            upload_status = gr.Textbox(
                label="Status",
                interactive=False,
                lines=1,
                placeholder="Upload status will appear here..."
            )

        with gr.Column(scale=2):
            gr.Markdown("### Ask a Question")
            query = gr.Textbox(
                label="Your Question",
                lines=3,
                placeholder="e.g. What are the main themes in this document?"
            )
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary", size="sm")
                submit_btn = gr.Button("Generate Answer", variant="primary", size="lg")
            answer = gr.Textbox(
                label="Answer",
                lines=12,
                interactive=False,
                placeholder="Your answer will appear here..."
            )

    gr.Markdown(
        "<div style='text-align:center; color:#555; font-size:0.85em; margin-top:20px;'>"
        "Powered by Llama 3 · ChromaDB · Sentence Transformers"
        "</div>"
    )

    upload_btn.click(fn=upload_file, inputs=file_input, outputs=upload_status)
    submit_btn.click(fn=query_fn, inputs=query, outputs=answer, show_progress=True)
    clear_btn.click(fn=lambda: ("", ""), outputs=[query, answer])

def gradio_app():
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    gradio_app()