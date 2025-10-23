# -*- coding: utf-8 -*-
"""
Hugging Face Spaces App Entry Point
Quantum LIMIT-Graph v2.4.0 NSN Integration Dashboard
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from huggingface_dashboard import create_gradio_interface

# Create and launch the dashboard
demo = create_gradio_interface()

if __name__ == '__main__':
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
