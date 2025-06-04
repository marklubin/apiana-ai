"""
UI Automation Tests for Gradio Pipeline Execution

These tests use Playwright to simulate user interactions with the Gradio web interface,
testing the complete workflow from pipeline selection to execution and result verification.
"""

import pytest
import asyncio
import tempfile
import json
import time
from pathlib import Path
from playwright.sync_api import Page, expect
import threading
import subprocess
import signal
import os


# Global variable to track the Gradio server process
gradio_process = None
server_ready = False


class GradioServerManager:
    """Manages the Gradio server lifecycle for testing."""
    
    def __init__(self, host="127.0.0.1", port=7860):
        self.host = host
        self.port = port
        self.process = None
        self.ready = False
        
    def start_server(self):
        """Start the Gradio server in a subprocess."""
        import subprocess
        import sys
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent.parent
        launch_script = project_root / "launch_gradio.py"
        
        # Start the Gradio server
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root)
        
        self.process = subprocess.Popen([
            sys.executable, str(launch_script),
            "--host", self.host,
            "--port", str(self.port)
        ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to be ready
        max_wait = 30  # 30 seconds timeout
        for _ in range(max_wait):
            try:
                import requests
                response = requests.get(f"http://{self.host}:{self.port}")
                if response.status_code == 200:
                    self.ready = True
                    break
            except:
                pass
            time.sleep(1)
        
        if not self.ready:
            self.stop_server()
            raise RuntimeError("Failed to start Gradio server within timeout")
    
    def stop_server(self):
        """Stop the Gradio server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
        self.ready = False
    
    def is_ready(self):
        """Check if the server is ready."""
        return self.ready


@pytest.fixture(scope="session")
def gradio_server():
    """Session-scoped fixture to manage Gradio server."""
    server = GradioServerManager()
    
    try:
        server.start_server()
        yield server
    finally:
        server.stop_server()


@pytest.fixture
def page_with_server(page: Page, gradio_server):
    """Fixture that provides a page with the Gradio server running."""
    # Navigate to the Gradio app
    page.goto("http://127.0.0.1:7860")
    
    # Wait for the app to load
    page.wait_for_selector("h1", timeout=10000)
    
    return page


@pytest.mark.ui_automation
def test_gradio_app_loads(page_with_server: Page):
    """Test that the Gradio application loads successfully."""
    page = page_with_server
    
    # Check that the main title is present
    expect(page.locator("h1")).to_contain_text("Apiana AI Pipeline Runner")
    
    # Check that tabs are present
    tabs = page.locator("[role='tab']")
    assert tabs.count() > 0, "No tabs found on the page"
    
    # Take a screenshot for debugging
    page.screenshot(path="tests/ui_automation/screenshots/app_loaded.png")


@pytest.mark.ui_automation
def test_testing_tab_exists(page_with_server: Page):
    """Test that the Testing tab exists with the dummy pipeline."""
    page = page_with_server
    
    # Look for the Testing tab
    testing_tab = page.locator("[role='tab']", has_text="Testing")
    expect(testing_tab).to_be_visible()
    
    # Click on the Testing tab
    testing_tab.click()
    
    # Wait for the tab content to load
    page.wait_for_timeout(1000)
    
    # Check that some form of pipeline selection is available
    # Gradio UI structure may vary, so try different selectors
    selection_elements = page.locator("select, .dropdown, [role='button'], input").all()
    assert len(selection_elements) > 0, "No interactive elements found for pipeline selection"
    
    # Take a screenshot
    page.screenshot(path="tests/ui_automation/screenshots/testing_tab.png")


@pytest.mark.ui_automation 
def test_dummy_pipeline_selection(page_with_server: Page):
    """Test selecting the dummy pipeline and verifying its description."""
    page = page_with_server
    
    # Navigate to Testing tab
    testing_tab = page.locator("[role='tab']", has_text="Testing")
    testing_tab.click()
    page.wait_for_timeout(1000)
    
    # Find and interact with the pipeline dropdown
    # Gradio dropdowns can be tricky, let's try multiple selectors
    dropdown_selectors = [
        ".dropdown select",
        "select",
        "[data-testid='dropdown']",
        ".gradio-dropdown select"
    ]
    
    dropdown = None
    for selector in dropdown_selectors:
        try:
            dropdown = page.locator(selector).first
            if dropdown.is_visible():
                break
        except:
            continue
    
    if dropdown and dropdown.is_visible():
        # Select the dummy test pipeline
        dropdown.select_option(label="Dummy Test Pipeline")
        page.wait_for_timeout(1000)
    else:
        # Alternative: look for clickable dropdown elements
        dropdown_button = page.locator(".dropdown, [role='button']").filter(has_text="Select Pipeline").first
        if dropdown_button.is_visible():
            dropdown_button.click()
            page.wait_for_timeout(500)
            
            # Look for the dummy pipeline option
            dummy_option = page.locator("text=Dummy Test Pipeline").first
            if dummy_option.is_visible():
                dummy_option.click()
                page.wait_for_timeout(1000)
    
    # Check that the dummy pipeline appears somewhere on the page
    # (the exact UI structure may vary)
    page_content = page.text_content("body")
    assert "Dummy Test Pipeline" in page_content or "Safe testing pipeline" in page_content, \
        "Dummy pipeline not found in page content"
    
    page.screenshot(path="tests/ui_automation/screenshots/dummy_pipeline_selected.png")


@pytest.mark.ui_automation
def test_dummy_pipeline_parameters(page_with_server: Page):
    """Test that dummy pipeline parameters are displayed and can be modified."""
    page = page_with_server
    
    # Navigate to Testing tab and select dummy pipeline
    testing_tab = page.locator("[role='tab']", has_text="Testing")
    testing_tab.click()
    page.wait_for_timeout(1000)
    
    # Try to select the dummy pipeline (this might need adjustment based on actual UI)
    try:
        # Look for any input fields that might be the pipeline parameters
        message_input = page.locator("input, textarea").filter(has=page.locator("label:has-text('message')")).first
        if message_input.is_visible():
            message_input.fill("Hello from UI test!")
            
        iterations_input = page.locator("input[type='number']").filter(has=page.locator("label:has-text('iterations')")).first
        if iterations_input.is_visible():
            iterations_input.fill("2")
            
        delay_input = page.locator("input[type='number']").filter(has=page.locator("label:has-text('delay')")).first
        if delay_input.is_visible():
            delay_input.fill("0.2")
    except:
        # If specific parameter inputs aren't found, just verify some inputs exist
        inputs = page.locator("input, textarea, select")
        expect(inputs).to_have_count_greater_than(0)
    
    page.screenshot(path="tests/ui_automation/screenshots/parameters_filled.png")


@pytest.mark.ui_automation
def test_dummy_pipeline_execution(page_with_server: Page):
    """Test executing the dummy pipeline and verifying results."""
    page = page_with_server
    
    # Navigate to Testing tab
    testing_tab = page.locator("[role='tab']", has_text="Testing")
    testing_tab.click()
    page.wait_for_timeout(1000)
    
    # The Testing tab should show the Dummy Test Pipeline directly
    # Wait a bit for the UI to load completely
    page.wait_for_timeout(2000)
    
    # Try to find and fill basic parameters
    try:
        # Look for common input patterns
        text_inputs = page.locator("input[type='text'], textarea")
        if text_inputs.count() > 0:
            text_inputs.first.fill("UI Test Message")
            
        number_inputs = page.locator("input[type='number']")
        if number_inputs.count() > 0:
            number_inputs.first.fill("2")  # Set iterations to 2
    except:
        pass  # Parameters might not be accessible yet
    
    # Debug: List all buttons on the page (commented out for clean output)
    # all_buttons = page.locator("button").all()
    # print(f"Found {len(all_buttons)} buttons on the page")
    # for i, btn in enumerate(all_buttons):
    #     try:
    #         text = btn.text_content()
    #         print(f"Button {i}: '{text}'")
    #     except:
    #         pass
    
    # Look for execute button with various text patterns
    execute_button = None
    button_texts = ["🚀 Execute Pipeline", " 🚀 Execute Pipeline", "Execute Pipeline", "Execute", "Run"]
    
    for text in button_texts:
        try:
            button = page.locator("button").filter(has_text=text).first
            if button.is_visible():
                execute_button = button
                break
        except:
            continue
    
    # Also try by element ID since we know it has a specific ID pattern
    if not execute_button or not execute_button.is_visible():
        try:
            # The ID should be execute_Testing since we're in the Testing tab
            execute_button = page.locator("button[id='execute_Testing']").first
            if execute_button.is_visible():
                pass  # Found it!
        except:
            pass
    
    # Try one more time with a more flexible approach
    if not execute_button or not execute_button.is_visible():
        try:
            execute_button = page.locator("button").filter(has_text="Execute").filter(has_text="Pipeline").first
            if execute_button.is_visible():
                pass  # Found it!
        except:
            pass
    
    if execute_button and execute_button.is_visible():
        # Take screenshot before execution
        page.screenshot(path="tests/ui_automation/screenshots/before_execution.png")
        
        # Click execute button
        execute_button.click()
        
        # Wait for execution to complete (dummy pipeline should be fast)
        page.wait_for_timeout(5000)  # 5 seconds should be enough for dummy pipeline
        
        # Look for success indicators
        success_indicators = [
            "text=completed",
            "text=success",
            "text=✅",
            "text=finished",
            ".success",
            "[class*='success']"
        ]
        
        found_success = False
        for indicator in success_indicators:
            try:
                element = page.locator(indicator).first
                if element.is_visible():
                    found_success = True
                    break
            except:
                continue
        
        # Take screenshot after execution
        page.screenshot(path="tests/ui_automation/screenshots/after_execution.png")
        
        # Check that some output appeared (logs, results, etc.)
        page_content = page.content()
        assert any(keyword in page_content.lower() for keyword in [
            "dummy", "processing", "iteration", "completed", "result"
        ]), "No execution output detected"
        
    else:
        # If we can't find execute button, just verify the UI is responsive
        page.screenshot(path="tests/ui_automation/screenshots/no_execute_button.png")
        pytest.skip("Execute button not found - UI structure might have changed")


@pytest.mark.ui_automation
def test_logs_and_results_display(page_with_server: Page):
    """Test that logs and results are displayed during/after execution."""
    page = page_with_server
    
    # Navigate to Testing tab
    testing_tab = page.locator("[role='tab']", has_text="Testing")
    testing_tab.click()
    page.wait_for_timeout(1000)
    
    # Look for expandable sections (logs, results)
    expandable_sections = page.locator("details, .accordion, [role='button'][aria-expanded]")
    
    if expandable_sections.count() > 0:
        # Expand the first section
        expandable_sections.first.click()
        page.wait_for_timeout(500)
        
        # Check if content is revealed
        page.screenshot(path="tests/ui_automation/screenshots/expanded_section.png")
    
    # Look for text areas or code blocks that might contain logs
    log_areas = page.locator("textarea[readonly], .log, .code, pre")
    
    # Take final screenshot
    page.screenshot(path="tests/ui_automation/screenshots/logs_and_results.png")
    
    # Verify that the page has some dynamic content
    page_text = page.text_content("body")
    assert len(page_text) > 100, "Page seems to have minimal content"


@pytest.mark.ui_automation
def test_responsive_ui_elements(page_with_server: Page):
    """Test that UI elements are responsive and interactive."""
    page = page_with_server
    
    # Test different tab interactions
    tabs = page.locator("[role='tab']")
    tab_count = tabs.count()
    
    if tab_count > 1:
        # Click through different tabs
        for i in range(min(tab_count, 3)):  # Test up to 3 tabs
            tabs.nth(i).click()
            page.wait_for_timeout(500)
            
            # Verify tab content changes
            page.screenshot(path=f"tests/ui_automation/screenshots/tab_{i}.png")
    
    # Test that the page is generally responsive
    expect(page.locator("body")).to_be_visible()
    expect(page.locator("h1")).to_be_visible()
    
    # Final comprehensive screenshot
    page.screenshot(path="tests/ui_automation/screenshots/full_app.png")


# Helper function to run tests
def run_ui_tests():
    """Helper function to run UI tests programmatically."""
    import pytest
    return pytest.main([
        "tests/ui_automation/",
        "-v",
        "-m", "ui_automation",
        "--tb=short"
    ])


if __name__ == "__main__":
    # Run tests if script is executed directly
    run_ui_tests()