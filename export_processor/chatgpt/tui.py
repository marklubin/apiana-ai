from textual.app import App
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import  Header, Static, DirectoryTree, LoadingIndicator, DataTable, TextArea
from textual.errors import RenderError
from textual.logging import TextualHandler
from .schema import OpenAIConversation
import json


class FilteredDirectoryTree(DirectoryTree):
    def filter_paths(self, paths):
        return [path for path in paths if not path.name.startswith(".")]
    
class FilePicker(Widget):
    path = reactive("../")
    
    def __init__(self, user_prompt, on_file_selected):
        super().__init__()
        self.user_prompt = user_prompt
        self.on_file_selected = on_file_selected
    
    def compose(self):
        yield Static(self.user_prompt)
        yield FilteredDirectoryTree(self.path)

    def on_directory_tree_directory_selected(self, event):
        self.path = event.path
        
    def on_directory_tree_file_selected(self, event):
        file_path = event.path
        if file_path:
            self.on_file_selected(file_path)

class ChatGPTExportProcessor(App):
    workflow_state  = reactive("new")
    input_filename = reactive(None)
    validated_convos = reactive([])
    prompt_text = reactive(None)
    
    def on_mount(self):
        self.theme = "solarized-light"
    
    def compose(self):
        yield Header("ChatGPT Export Processor")
        
        if self.workflow_state == "new":
            yield FilePicker("Select a file to process ChatGPT conversations:",
                             self.on_file_selection_complete)
        elif self.workflow_state == "loading":
            yield LoadingIndicator("Processing file...")
        elif self.workflow_state == "loaded":
            yield Static(f"Processed {len(self.validated_convos)} conversations.")
            dt = DataTable()
            self.load_table(dt)
            yield dt
            yield Static("Press ENTER to select summarization prompt.")
        elif self.workflow_state == "prompt_selection":
            yield FilePicker(
                "Select a summarization prompt file:",
                self.on_prompt_file_selection_complete
            )
        elif self.workflow_state == "comfirming":
            yield Static(f"Ready to summarize {len(self.validated_convos)} conversations selected Prompt.")
            yield TextArea(text=self.prompt_text)
            
            
    def key_enter(self):
        if self.workflow_state == "loaded":
            self.change_workflow_state("prompt_selection")
        
    def on_prompt_file_selection_complete(self, file_path: str) -> None:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.prompt_text = file.read()
                self.change_workflow_state("comfirming")
        except Exception as e:
            print(f"❌ Error reading prompt file: {e}")
            self.log.error(f"Error reading prompt file: {e}")
            raise RenderError(f"Error reading prompt file: {e}", e)
            
    def on_file_selection_complete(self, file_path: str) -> None:
        self.input_filename = file_path
        self.change_workflow_state("loading")
        self.import_file(self.input_filename)
        self.change_workflow_state("loaded")
        
    def import_file(self, file_path: str) -> None:
        successful_imports = 0
        failed_imports = 0
        
        try:
            print(f"🔍 Starting to process file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as file:
                conversations = json.load(file)
                
            if not isinstance(conversations, list):
                print("❌ Invalid file format: 'conversations' is not a list")
                self.log.error("Invalid file format: 'conversations' is not a list")
                return
                
            print(f"🔍 Found {len(conversations)} conversations to process")
            
            for i, convo_data in enumerate(conversations):
                try:
                    validated_convo = OpenAIConversation.from_dict(convo_data)
                    if validated_convo:
                        self.validated_convos.append(validated_convo)
                        successful_imports += 1
                        print(f"✅ Processed conversation {i}: '{validated_convo.title}' ({len(validated_convo.messages)} messages)")
                    else:
                        failed_imports += 1
                        print(f"⚠️  Skipped conversation {i}: no valid content")
                        
                except Exception as e:
                    failed_imports += 1
                    print(f"❌ Failed to process conversation {i}: {e}")
                    self.log.error(f"Failed to process conversation {i}: {e}")
                
        except FileNotFoundError:
            error_msg = f"File not found: {file_path}"
            print(f"❌ {error_msg}")
            self.log.error(error_msg)
            return
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in file {file_path}: {e}"
            print(f"❌ {error_msg}")
            self.log.error(error_msg)
            return
        except Exception as e:
            error_msg = f"Unexpected error processing file {file_path}: {e}"
            print(f"❌ {error_msg}")
            self.log.error(error_msg)
            return
        
        print(f"🎉 Import complete: {successful_imports} successful, {failed_imports} failed")
        self.log.info(f"Import complete: {successful_imports} successful, {failed_imports} failed")
        
    def load_table(self, data_table: DataTable) -> None:
        data_table.clear()
        data_table.add_column("Title")
        data_table.add_column("Number of Messages")

        data_table.add_rows([
            (convo.title, len(convo.messages))
            for convo in self.validated_convos
        ])
        
    def change_workflow_state(self, new_state: str) -> None:
        self.workflow_state = new_state
        self.refresh(recompose=True)
 
 
def main():
    ChatGPTExportProcessor().run()
         
if __name__ == "__main__":
    main()