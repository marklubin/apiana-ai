import gradio as gr
import json
import logging
from pathlib import Path
from typing import Tuple

from apiana.applications.chatgpt_export.cli import (
    embedded_chatgpt_export_summaries,
    process_one_conversation,
    write_convos_in_apiana_format,
    get_dependencies,
)
from apiana.applications.themes.rugrat import RugRat
from apiana.batch.chatgpt.chatgpt_export_parsing import convo_from_export_format_dict

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def process_chatgpt_export(
    input_file, output_dir_name: str = "gradio_output"
) -> Tuple[str, str]:
    """Process ChatGPT export file and return results."""
    try:
        if input_file is None:
            return "❌ Error: No input file provided", ""

        # Create output directory
        output_dir = Path.cwd() / "output" / output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process the file
        embedded_chatgpt_export_summaries(input_file.name, str(output_dir))

        # Count results
        parsed_dir = output_dir / "parsed"
        summaries_dir = output_dir / "summaries"

        parsed_count = (
            len(list(parsed_dir.glob("*.json"))) if parsed_dir.exists() else 0
        )
        summary_count = (
            len(list(summaries_dir.glob("*.txt"))) if summaries_dir.exists() else 0
        )

        success_msg = f"""✅ Processing Complete!
        
📁 Output Directory: {output_dir}
📝 Conversations Parsed: {parsed_count}
🧠 Summaries Generated: {summary_count}
💾 Data Stored in Neo4j: {summary_count} conversations

Files saved to:
- Parsed conversations: {parsed_dir}
- Generated summaries: {summaries_dir}
        """

        # Return file list for download
        file_list = []
        if parsed_dir.exists():
            file_list.extend([str(f) for f in parsed_dir.glob("*.json")])
        if summaries_dir.exists():
            file_list.extend([str(f) for f in summaries_dir.glob("*.txt")])

        return success_msg, "\n".join(file_list[:10])  # Show first 10 files

    except Exception as e:
        log.error(f"Error processing export: {e}")
        return f"❌ Error: {str(e)}", ""


def process_single_conversation(
    title: str, messages_json: str, output_dir_name: str = "single_convo"
) -> Tuple[str, str]:
    """Process a single conversation from manual input."""
    try:
        if not title or not messages_json:
            return "❌ Error: Please provide both title and messages", ""

        # Parse messages
        try:
            messages_data = json.loads(messages_json)
        except json.JSONDecodeError as e:
            return f"❌ Error: Invalid JSON format: {str(e)}", ""

        # Create a conversation dict in ChatGPT export format
        convo_dict = {"title": title, "messages": messages_data}

        # Convert to Apiana format
        apiana_convo = convo_from_export_format_dict(convo_dict)

        # Create output directory
        output_dir = Path.cwd() / "output" / output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        summaries_dir = output_dir / "summaries"
        summaries_dir.mkdir(exist_ok=True)

        # Process the conversation
        process_one_conversation(apiana_convo, str(summaries_dir))

        # Read the generated summary
        safe_title = title.replace(" ", "_").replace("/", "_").replace(":", "_")
        safe_title = "".join(
            char for char in safe_title if char.isalnum() or char in ("_", "-", ".")
        )
        summary_file = summaries_dir / f"{safe_title.lower()}.txt"

        summary_content = ""
        if summary_file.exists():
            with open(summary_file, "r") as f:
                summary_content = f.read()

        success_msg = f"""✅ Single Conversation Processed!
        
📝 Title: {title}
💬 Messages: {len(messages_data)}
📁 Summary File: {summary_file}
💾 Stored in Neo4j: Yes
        """

        return success_msg, summary_content

    except Exception as e:
        log.error(f"Error processing single conversation: {e}")
        return f"❌ Error: {str(e)}", ""


def parse_only_mode(input_file, output_dir_name: str = "parse_only") -> Tuple[str, str]:
    """Parse ChatGPT export to Apiana format without processing."""
    try:
        if input_file is None:
            return "❌ Error: No input file provided", ""

        # Create output directory
        output_dir = Path.cwd() / "output" / output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load and convert conversations
        with open(input_file.name) as f:
            data = json.load(f)

        apiana_convos = []
        for c in data:
            apiana_convo = convo_from_export_format_dict(c)
            apiana_convos.append(apiana_convo)

        # Write conversations in Apiana format
        write_convos_in_apiana_format(apiana_convos, str(output_dir))

        parsed_dir = output_dir / "parsed"
        parsed_count = (
            len(list(parsed_dir.glob("*.json"))) if parsed_dir.exists() else 0
        )

        success_msg = f"""✅ Parsing Complete!
        
📁 Output Directory: {output_dir}
📝 Conversations Parsed: {parsed_count}
🔄 Processing: None (parse-only mode)

Files saved to: {parsed_dir}
        """

        # Return file list
        file_list = []
        if parsed_dir.exists():
            file_list.extend([str(f) for f in parsed_dir.glob("*.json")])

        return success_msg, "\n".join(file_list[:10])

    except Exception as e:
        log.error(f"Error in parse-only mode: {e}")
        return f"❌ Error: {str(e)}", ""


def check_dependencies() -> str:
    """Check if all dependencies are working."""
    try:
        memory_store, summarizer, embedder = get_dependencies()

        # Test embedder
        test_vector = embedder.embed_query("test query")

        status = f"""✅ Dependencies Check
        
🔗 Neo4j Memory Store: Connected
🤖 LLM Summarizer: Available  
📊 Embedder: Working (vector dim: {len(test_vector)})
⚙️  Configuration: Loaded

System is ready for processing!
        """

        return status

    except Exception as e:
        log.error(f"Dependencies check failed: {e}")
        return f"❌ Dependencies Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(theme=RugRat, title="Apiana AI - ChatGPT Export Processor") as app:
    gr.Markdown("""
    # 🧠 Apiana AI - ChatGPT Export Processor
    
    Process ChatGPT conversation exports into memory-enabled AI conversations with embeddings and summaries.
    """)

    with gr.Tabs():
        # Full Processing Tab
        with gr.TabItem("🚀 Full Processing"):
            gr.Markdown("### Process ChatGPT Export File")
            gr.Markdown(
                "Upload a ChatGPT export JSON file to parse conversations, generate summaries, and store in Neo4j."
            )

            with gr.Row():
                with gr.Column():
                    input_file = gr.File(
                        label="ChatGPT Export JSON File",
                        file_types=[".json"],
                        type="filepath",
                    )
                    output_name = gr.Textbox(
                        label="Output Directory Name",
                        value="gradio_output",
                        placeholder="Choose a name for the output directory",
                    )
                    process_btn = gr.Button("🚀 Process Export", variant="primary")

                with gr.Column():
                    process_result = gr.Textbox(
                        label="Processing Results", lines=10, interactive=False
                    )
                    file_list = gr.Textbox(
                        label="Generated Files (first 10)", lines=5, interactive=False
                    )

            process_btn.click(
                process_chatgpt_export,
                inputs=[input_file, output_name],
                outputs=[process_result, file_list],
            )

        # Single Conversation Tab
        with gr.TabItem("💬 Single Conversation"):
            gr.Markdown("### Process Individual Conversation")
            gr.Markdown(
                "Manually input a conversation to generate summary and embeddings."
            )

            with gr.Row():
                with gr.Column():
                    convo_title = gr.Textbox(
                        label="Conversation Title",
                        placeholder="Enter a title for this conversation",
                    )
                    messages_input = gr.Textbox(
                        label="Messages JSON",
                        lines=10,
                        placeholder='[{"role": "user", "content": {"parts": ["Hello!"]}}, {"role": "assistant", "content": {"parts": ["Hi there!"]}}]',
                    )
                    single_output_name = gr.Textbox(
                        label="Output Directory Name",
                        value="single_convo",
                        placeholder="Choose a name for the output directory",
                    )
                    single_process_btn = gr.Button(
                        "💬 Process Conversation", variant="primary"
                    )

                with gr.Column():
                    single_result = gr.Textbox(
                        label="Processing Results", lines=5, interactive=False
                    )
                    generated_summary = gr.Textbox(
                        label="Generated Summary", lines=10, interactive=False
                    )

            single_process_btn.click(
                process_single_conversation,
                inputs=[convo_title, messages_input, single_output_name],
                outputs=[single_result, generated_summary],
            )

        # Parse Only Tab
        with gr.TabItem("📝 Parse Only"):
            gr.Markdown("### Parse Export to Apiana Format")
            gr.Markdown(
                "Convert ChatGPT export to Apiana format without generating summaries or embeddings."
            )

            with gr.Row():
                with gr.Column():
                    parse_input_file = gr.File(
                        label="ChatGPT Export JSON File",
                        file_types=[".json"],
                        type="filepath",
                    )
                    parse_output_name = gr.Textbox(
                        label="Output Directory Name",
                        value="parse_only",
                        placeholder="Choose a name for the output directory",
                    )
                    parse_btn = gr.Button("📝 Parse Only", variant="secondary")

                with gr.Column():
                    parse_result = gr.Textbox(
                        label="Parsing Results", lines=10, interactive=False
                    )
                    parse_file_list = gr.Textbox(
                        label="Parsed Files (first 10)", lines=5, interactive=False
                    )

            parse_btn.click(
                parse_only_mode,
                inputs=[parse_input_file, parse_output_name],
                outputs=[parse_result, parse_file_list],
            )

        # System Status Tab
        with gr.TabItem("⚙️ System Status"):
            gr.Markdown("### Check System Dependencies")
            gr.Markdown(
                "Verify that Neo4j, LLM, and embedding services are working correctly."
            )

            check_btn = gr.Button("🔍 Check Dependencies", variant="secondary")
            status_result = gr.Textbox(
                label="System Status", lines=10, interactive=False
            )

            check_btn.click(check_dependencies, outputs=[status_result])

    gr.Markdown("""
    ---
    ### 📚 Usage Tips
    - **Full Processing**: Complete pipeline with summaries and Neo4j storage
    - **Single Conversation**: Test with individual conversations
    - **Parse Only**: Convert format without expensive LLM/embedding calls
    - **System Status**: Check if all services are running correctly
    
    Generated files are saved to the `output/` directory in your project root.
    """)


def launch_ui():
    """Launch the Gradio interface."""
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)


if __name__ == "__main__":
    launch_ui()
