#!/usr/bin/env python3
"""
Launch the Gradio UI for Apiana AI ChatGPT Export Processor
"""

from apiana.applications.gradio_ui import launch_ui

if __name__ == "__main__":
    print("🚀 Launching Apiana AI Gradio Interface...")
    print("📍 URL: http://localhost:7860")
    print("🔧 Use Ctrl+C to stop the server")
    print("")
    launch_ui()