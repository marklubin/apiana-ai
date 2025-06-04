"""
Main Gradio Application

Provides a web interface for discovering, configuring, and executing
pipelines with real-time progress tracking and log display.
"""

import gradio as gr
from typing import List
import threading
import time
import logging
from datetime import datetime

from apiana.applications.gradio.pipeline_discovery import get_pipeline_discovery
from apiana.applications.gradio.ui_components import PipelineUI, PipelineExecutor


class GradioApp:
    """Main Gradio application for pipeline execution."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 7860):
        """
        Initialize the Gradio application.
        
        Args:
            host: Host address to bind to
            port: Port to bind to
        """
        self.host = host
        self.port = port
        self.discovery = get_pipeline_discovery()
        self.ui = PipelineUI()
        self.executor = PipelineExecutor()
        self.interface = None
        
        # Setup logging for the application
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface."""
        
        with gr.Blocks(
            theme=self.ui.create_custom_theme(),
            title="Apiana AI Pipeline Runner",
            css=self._get_custom_css()
        ) as interface:
            
            # Header
            gr.Markdown("""
            # 🚀 Apiana AI Pipeline Runner
            
            Automatically discover and execute processing pipelines with real-time monitoring.
            """)
            
            # Global state for current execution
            execution_state = gr.State({
                "pipeline_name": None,
                "parameters": {},
                "status": "idle",
                "logs": "",
                "progress": 0
            })
            
            with gr.Tabs():
                
                # Pipeline Categories
                categories = self.discovery.get_pipelines_by_category()
                
                for category, pipeline_names in categories.items():
                    with gr.TabItem(category, id=f"tab_{category.lower().replace(' ', '_')}"):
                        
                        # Pipeline selection section
                        with gr.Group():
                            gr.Markdown(f"### {category} Pipelines")
                            
                            pipeline_dropdown = gr.Dropdown(
                                choices=pipeline_names,
                                label="Select Pipeline",
                                value=pipeline_names[0] if pipeline_names else None,
                                elem_id=f"pipeline_dropdown_{category}"
                            )
                            
                            pipeline_description = gr.Markdown(
                                "Select a pipeline to see details...",
                                elem_id=f"description_{category}"
                            )
                        
                        # Configuration section
                        with gr.Group():
                            gr.Markdown("### Configuration")
                            
                            # Neo4j configuration (common to most pipelines)
                            with gr.Accordion("Neo4j Database Configuration", open=True) as neo4j_accordion:
                                with gr.Row():
                                    neo4j_username = gr.Textbox(
                                        label="Username",
                                        value="neo4j",
                                        elem_id=f"neo4j_user_{category}"
                                    )
                                    neo4j_password = gr.Textbox(
                                        label="Password",
                                        type="password",
                                        elem_id=f"neo4j_pass_{category}"
                                    )
                                
                                with gr.Row():
                                    neo4j_host = gr.Textbox(
                                        label="Host",
                                        value="localhost",
                                        elem_id=f"neo4j_host_{category}"
                                    )
                                    neo4j_port = gr.Number(
                                        label="Port",
                                        value=7687,
                                        precision=0,
                                        elem_id=f"neo4j_port_{category}"
                                    )
                            
                            # Dynamic parameter inputs
                            parameter_container = gr.Column(
                                elem_id=f"params_{category}",
                                visible=False
                            )
                        
                        # Execution section
                        with gr.Group():
                            gr.Markdown("### Execution")
                            
                            with gr.Row():
                                execute_btn = gr.Button(
                                    "🚀 Execute Pipeline",
                                    variant="primary",
                                    size="lg",
                                    elem_id=f"execute_{category}"
                                )
                                stop_btn = gr.Button(
                                    "⏹️ Stop",
                                    variant="stop",
                                    elem_id=f"stop_{category}"
                                )
                                clear_logs_btn = gr.Button(
                                    "🗑️ Clear Logs",
                                    variant="secondary",
                                    elem_id=f"clear_{category}"
                                )
                        
                        # Status and progress section
                        with gr.Group():
                            gr.Markdown("### Status & Progress")
                            
                            with gr.Row():
                                status_display = gr.Textbox(
                                    label="Status",
                                    value="Ready",
                                    interactive=False,
                                    elem_id=f"status_{category}"
                                )
                                progress_display = gr.Textbox(
                                    label="Progress",
                                    value="0%",
                                    interactive=False,
                                    elem_id=f"progress_{category}"
                                )
                            
                            # Pipeline steps visualization
                            steps_display = gr.Markdown(
                                "### Pipeline Steps\n*Select a pipeline to see processing steps*",
                                elem_id=f"steps_{category}"
                            )
                        
                        # Logs section
                        with gr.Accordion("📋 Execution Logs", open=False):
                            logs_display = gr.Textbox(
                                label="",
                                value="",
                                lines=20,
                                max_lines=50,
                                interactive=False,
                                container=False,
                                elem_classes=["log-display"],
                                elem_id=f"logs_{category}"
                            )
                        
                        # Results section
                        with gr.Accordion("📊 Results", open=False):
                            results_display = gr.JSON(
                                label="Pipeline Results",
                                elem_id=f"results_{category}"
                            )
                        
                        # Event handlers for this category tab
                        self._setup_tab_events(
                            category,
                            pipeline_dropdown,
                            pipeline_description,
                            parameter_container,
                            neo4j_accordion,
                            execute_btn,
                            stop_btn,
                            clear_logs_btn,
                            status_display,
                            progress_display,
                            steps_display,
                            logs_display,
                            results_display,
                            execution_state,
                            [neo4j_username, neo4j_password, neo4j_host, neo4j_port]
                        )
                
                # About tab
                with gr.TabItem("ℹ️ About"):
                    gr.Markdown("""
                    ## About Apiana AI Pipeline Runner
                    
                    This application automatically discovers pipeline factory functions and provides
                    a dynamic web interface for configuration and execution.
                    
                    ### Features
                    - **Automatic Discovery**: Finds all available pipeline factories
                    - **Dynamic UI**: Generates forms based on pipeline parameters
                    - **Real-time Monitoring**: Live progress tracking and log display
                    - **Multi-threading**: Non-blocking pipeline execution
                    - **Custom Styling**: Beautiful interface with custom theme
                    
                    ### Available Pipelines
                    """)
                    
                    # Show discovered pipelines
                    pipeline_info = []
                    for category, pipelines in categories.items():
                        pipeline_info.append(f"**{category}:**")
                        for pipeline_name in pipelines:
                            metadata = self.discovery.get_pipeline_metadata(pipeline_name)
                            pipeline_info.append(f"- {metadata.name}: {metadata.description}")
                        pipeline_info.append("")
                    
                    gr.Markdown("\n".join(pipeline_info))
        
        return interface
    
    def _setup_tab_events(
        self,
        category: str,
        pipeline_dropdown: gr.Dropdown,
        pipeline_description: gr.Markdown,
        parameter_container: gr.Column,
        neo4j_accordion: gr.Accordion,
        execute_btn: gr.Button,
        stop_btn: gr.Button,
        clear_logs_btn: gr.Button,
        status_display: gr.Textbox,
        progress_display: gr.Textbox,
        steps_display: gr.Markdown,
        logs_display: gr.Textbox,
        results_display: gr.JSON,
        execution_state: gr.State,
        neo4j_inputs: List[gr.Component]
    ):
        """Setup event handlers for a category tab."""
        
        # Dynamic parameter components storage (for future use)
        # dynamic_components = {}
        
        def on_pipeline_select(pipeline_name, current_state):
            """Handle pipeline selection."""
            if not pipeline_name:
                return (
                    "Select a pipeline to see details...",
                    gr.update(visible=False),
                    gr.update(value="*Select a pipeline to see processing steps*"),
                    {},  # Clear dynamic components
                    current_state
                )
            
            metadata = self.discovery.get_pipeline_metadata(pipeline_name)
            # parameters = self.discovery.get_pipeline_parameters(pipeline_name)
            
            # Update description
            description_md = f"""
            ### {metadata.name}
            
            **Description:** {metadata.description}
            
            **Category:** {metadata.category}
            
            **Estimated Time:** {metadata.estimated_time}
            
            **Output:** {metadata.output_description}
            
            **Tags:** {', '.join(metadata.tags)}
            
            **File Upload Required:** {'Yes' if metadata.requires_files else 'No'}
            """
            
            # Create pipeline steps visualization
            steps_md = "### Pipeline Steps\n\n"
            # In a real implementation, this would inspect the actual pipeline
            example_steps = [
                ("🔍", "Input Validation", "Validate input parameters and files"),
                ("📖", "Data Reading", "Read and parse input data"),
                ("💾", "Data Storage", "Store processed data in database"),
                ("🧠", "AI Processing", "Apply ML models and transformations"),
                ("✅", "Completion", "Finalize and save results")
            ]
            
            for i, (icon, step_name, step_desc) in enumerate(example_steps, 1):
                steps_md += f"{i}. {icon} **{step_name}**\n   *{step_desc}*\n\n"
            
            # Update state
            new_state = current_state.copy()
            new_state.update({
                "pipeline_name": pipeline_name,
                "parameters": {},
                "status": "configured"
            })
            
            return (
                description_md,
                gr.update(visible=True),
                steps_md,
                {},  # Dynamic components would be updated here
                new_state
            )
        
        def execute_pipeline_fn(current_state, *neo4j_values):
            """Execute the selected pipeline."""
            pipeline_name = current_state.get("pipeline_name")
            if not pipeline_name:
                return (
                    gr.update(value="No pipeline selected"),
                    gr.update(value="Error"),
                    gr.update(value=""),
                    gr.update(value={}),
                    current_state
                )
            
            # Prepare Neo4j config (for future use)
            # neo4j_config = Neo4jConfig(
            #     username=neo4j_values[0],
            #     password=neo4j_values[1],
            #     host=neo4j_values[2],
            #     port=int(neo4j_values[3]),
            #     database="neo4j"
            # )
            
            # Start execution in thread
            def run_execution():
                try:
                    # This is a simplified example
                    # Real implementation would use self.executor
                    time.sleep(1)  # Simulate work
                    self.logger.info(f"Starting execution of {pipeline_name}")
                    time.sleep(2)
                    self.logger.info("Pipeline execution completed successfully")
                except Exception as e:
                    self.logger.error(f"Pipeline execution failed: {e}")
            
            thread = threading.Thread(target=run_execution)
            thread.start()
            
            new_state = current_state.copy()
            new_state["status"] = "running"
            
            return (
                gr.update(value="Pipeline execution started..."),
                gr.update(value="Running"),
                gr.update(value="Starting pipeline execution..."),
                gr.update(value={"status": "started", "timestamp": str(datetime.now())}),
                new_state
            )
        
        def clear_logs_fn():
            """Clear the logs display."""
            return gr.update(value="")
        
        def update_logs():
            """Update logs display (called periodically)."""
            return gr.update(value=self.executor.get_status().get("logs", ""))
        
        # Connect events
        pipeline_dropdown.change(
            on_pipeline_select,
            inputs=[pipeline_dropdown, execution_state],
            outputs=[
                pipeline_description,
                parameter_container,
                steps_display,
                gr.State({}),  # Dynamic components placeholder
                execution_state
            ]
        )
        
        execute_btn.click(
            execute_pipeline_fn,
            inputs=[execution_state] + neo4j_inputs,
            outputs=[
                logs_display,
                status_display,
                progress_display,
                results_display,
                execution_state
            ]
        )
        
        clear_logs_btn.click(
            clear_logs_fn,
            outputs=[logs_display]
        )
    
    def _get_custom_css(self) -> str:
        """Get custom CSS for styling."""
        return """
        .log-display {
            font-family: 'Courier New', Monaco, Consolas, monospace !important;
            background-color: #FFFBEB !important;
            border: 1px solid #D4AF37 !important;
            font-size: 12px !important;
        }
        
        .gradio-container {
            font-family: 'Courier New', Monaco, Consolas, monospace !important;
        }
        
        .progress-bar {
            background-color: #10B981 !important;
        }
        
        .status-running {
            background-color: #FEF3C7 !important;
            color: #92400E !important;
        }
        
        .status-completed {
            background-color: #D1FAE5 !important;
            color: #065F46 !important;
        }
        
        .status-error {
            background-color: #FEE2E2 !important;
            color: #991B1B !important;
        }
        """
    
    def launch(self, share: bool = False, debug: bool = False):
        """
        Launch the Gradio application.
        
        Args:
            share: Whether to create a public link
            debug: Whether to enable debug mode
        """
        self.interface = self.create_interface()
        
        self.logger.info(f"Starting Gradio application on {self.host}:{self.port}")
        
        self.interface.launch(
            server_name=self.host,
            server_port=self.port,
            share=share,
            debug=debug,
            show_error=True,
            quiet=not debug
        )
    
    def close(self):
        """Close the Gradio application."""
        if self.interface:
            self.interface.close()
            self.logger.info("Gradio application closed")


def create_app(host: str = "127.0.0.1", port: int = 7860) -> GradioApp:
    """
    Create a new Gradio application instance.
    
    Args:
        host: Host address to bind to
        port: Port to bind to
        
    Returns:
        GradioApp instance
    """
    return GradioApp(host=host, port=port)