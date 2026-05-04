import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from app import gradio_app

if __name__ == "__main__":
    gradio_app()