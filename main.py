from export_processor.chatgpt.tui import ChatGPTExportProcessor
import debugpy
if __name__ == "__main__":
    
  # Wait for the debugger to attach
    ChatGPTExportProcessor().run()
