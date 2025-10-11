import reflex as rx
from pathlib import Path

# Path to the judge outputs folder (under project root)
JUDGE_OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "judge_outputs"
FEEDBACK_STORAGE_DIR = Path(__file__).resolve().parent / "feedback_storage"
FEEDBACK_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


class FeedbackState(rx.State):
    """App state for managing judge outputs and feedback."""

    selected_file: str = ""
    file_content: str = ""
    feedback_text: str = ""
    saved_message: str = ""

    def load_file(self, filename: str):
        """Load and display the selected file."""
        file_path = JUDGE_OUTPUTS_DIR / filename
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                self.file_content = f.read()
            self.selected_file = filename
            self.feedback_text = ""
            self.saved_message = f"Loaded: {filename}"
        else:
            self.file_content = ""
            self.saved_message = "File not found."

    def save_feedback(self):
        """Save feedback to a file inside feedback_storage."""
        if not self.selected_file:
            self.saved_message = " Please select a file first."
            return

        feedback_file = FEEDBACK_STORAGE_DIR / f"{self.selected_file}_feedback.txt"
        with open(feedback_file, "w", encoding="utf-8") as f:
            f.write(self.feedback_text.strip() or "(No feedback provided)")

        self.saved_message = f" Feedback saved for {self.selected_file}"

    @rx.var
    def judge_output_files(self) -> list[str]:
        """List available judge output .py files."""
        if JUDGE_OUTPUTS_DIR.exists():
            return sorted([f.name for f in JUDGE_OUTPUTS_DIR.glob("*.json")])
        return []


def feedback_page() -> rx.Component:
    """Main human feedback page layout."""
    return rx.container(
        rx.vstack(
            rx.heading(" LLM Judge Human Feedback Dashboard", size="7"),
            rx.text("Review and provide feedback for each LLM-as-a-Judge output file.", size="4"),
            rx.divider(),

            # Dropdown for file selection
            rx.hstack(
                rx.text("Select a judge output file:", size="3"),
                rx.select(
                    FeedbackState.judge_output_files,
                    placeholder="Choose file...",
                    value=FeedbackState.selected_file,
                    on_change=FeedbackState.load_file,
                    width="50%",
                ),
            ),

            # File content display
            rx.box(
                rx.text("File Content Preview:", size="4", weight="bold"),
                rx.text_area(
                    value=FeedbackState.file_content,
                    read_only=True,
                    min_height="300px",
                    width="100%",
                    resize="vertical",
                ),
                padding_y="1em",
            ),

            # Feedback input area
            rx.box(
                rx.text("Your Feedback:", size="4", weight="bold"),
                rx.text_area(
                    placeholder="Enter your feedback or corrections here...",
                    value=FeedbackState.feedback_text,
                    on_change=lambda v: FeedbackState.set_feedback_text(v),
                    min_height="150px",
                    width="100%",
                    resize="vertical",
                ),
                padding_y="1em",
            ),

            # Save button + message
            rx.hstack(
                rx.button(
                    "Save Feedback",
                    on_click=FeedbackState.save_feedback,
                    color_scheme="blue",
                    size="4",
                ),
                rx.text(FeedbackState.saved_message, size="3", color="green"),
                spacing="4",
                align="center",
            ),

            spacing="6",
            padding="2em",
            align="start",
        ),
        width="100%",
        padding="2em",
    )


# Reflex entry point
app = rx.App()
app.add_page(
    feedback_page,
    route="/", 
    title="Human Feedback | LLM Judge Review"
)