import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate import generate_answer
from ingest import ingest_documents

DATA_FOLDER = "data"

def upload_file(file):
    if file is None:
        return "No file uploaded."
    
    filename = os.path.basename(file.name)
    dest = os.path.join(DATA_FOLDER, filename)
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


def gradio_app():
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio is not installed. Install it with `pip install gradio`.")

    with gr.Blocks(
        title="RAG Q&A Bot",
        css="""
            /* ── Layout ── */
            .header { text-align: center; padding: 20px 0 5px 0; }
            .subheader { text-align: center; color: #888; margin-bottom: 20px; }
            .upload-box { border: 1px solid #333; border-radius: 10px; padding: 15px; }
            .qa-box { border: 1px solid #333; border-radius: 10px; padding: 15px; }
            footer { display: none !important; }

            /* ── Primary button (Generate Answer) ── */
            .primary-btn button {
                transition: all 0.2s ease;
                box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            }
            .primary-btn button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 16px rgba(99, 102, 241, 0.5);
                filter: brightness(1.1);
            }
            .primary-btn button:active {
                transform: translateY(0px);
                box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            }

            /* ── Secondary button (Upload & Clear) ── */
            .secondary-btn button {
                transition: all 0.2s ease;
                box-shadow: 0 2px 4px rgba(0,0,0,0.15);
            }
            .secondary-btn button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.25);
                filter: brightness(1.15);
            }
            .secondary-btn button:active {
                transform: translateY(0px);
                filter: brightness(0.95);
            }
        """
    ) as demo:

        # ── Header ──────────────────────────────────────
        gr.Markdown("# RAG Q&A Bot", elem_classes="header")
        gr.Markdown(
            "Upload a document and ask questions. Answers are grounded in your content.",
            elem_classes="subheader"
        )

        with gr.Row():
            # ── Upload Section ───────────────────────────
            with gr.Column(scale=1, elem_classes="upload-box"):
                gr.Markdown("### Upload Document")
                gr.Markdown("Supported formats: PDF, TXT, DOCX")
                file_input = gr.File(
                    label="Choose a file",
                    file_types=[".pdf", ".txt", ".docx"]
                )
                upload_btn = gr.Button(
                    "Upload & Ingest",
                    variant="secondary",
                    size="lg",
                    elem_classes="secondary-btn"
                )
                upload_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=1,
                    placeholder="Upload status will appear here..."
                )

            # ── Q&A Section ──────────────────────────────
            with gr.Column(scale=2, elem_classes="qa-box"):
                gr.Markdown("### Ask a Question")
                query = gr.Textbox(
                    label="Your Question",
                    lines=3,
                    placeholder="e.g. What are the main themes in this document?"
                )
                with gr.Row():
                    clear_btn = gr.Button(
                        "Clear",
                        variant="secondary",
                        size="sm",
                        elem_classes="secondary-btn"
                    )
                    submit_btn = gr.Button(
                        "Generate Answer",
                        variant="primary",
                        size="lg",
                        elem_classes="primary-btn"
                    )
                answer = gr.Textbox(
                    label="Answer",
                    lines=12,
                    interactive=False,
                    placeholder="Your answer will appear here..."
                )

        # ── Footer ───────────────────────────────────────
        gr.Markdown(
            "<div style='text-align:center; color:#555; font-size:0.85em; margin-top:20px;'>"
            "Powered by Llama 3 · ChromaDB · Sentence Transformers"
            "</div>"
        )

        # ── Events ───────────────────────────────────────
        upload_btn.click(fn=upload_file, inputs=file_input, outputs=upload_status)
        submit_btn.click(fn=query_fn, inputs=query, outputs=answer, show_progress=True)
        clear_btn.click(fn=lambda: ("", ""), outputs=[query, answer])

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    gradio_app()