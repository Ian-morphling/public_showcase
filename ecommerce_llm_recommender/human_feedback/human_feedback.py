import reflex as rx
from pathlib import Path
import json

# Paths
JUDGE_OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "judge_outputs"
FEEDBACK_STORAGE_DIR = Path(__file__).resolve().parent / "feedback_storage"
FEEDBACK_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


class FeedbackState(rx.State):
    """App state for managing judge outputs and human feedback."""

    selected_file: str = ""
    file_content: str = ""
    feedback_text: str = ""
    saved_message: str = ""
    query: str = ""
    answer: str = ""
    judge_scores: dict = {}
    human_scores: dict = {}
    docs_used: list[str] = []

    # Toggle states
    show_llm_analysis: bool = True
    show_query_results: bool = True

    @rx.var
    def judge_output_files(self) -> list[str]:
        if JUDGE_OUTPUTS_DIR.exists():
            return sorted([f.name for f in JUDGE_OUTPUTS_DIR.glob("*.json")])
        return []

    @rx.var
    def human_score_display(self) -> dict:
        return {metric: str(self.human_scores.get(metric, "N/A"))
                for metric in ["Relevance", "Groundedness", "Balance"]}

    @rx.var
    def analysis_button_label(self) -> str:
        return "Hide LLM Analysis" if self.show_llm_analysis else "Show LLM Analysis"

    @rx.var
    def query_results_button_label(self) -> str:
        return "Hide Query Results" if self.show_query_results else "Show Query Results"

    def toggle_llm_analysis(self):
        self.show_llm_analysis = not self.show_llm_analysis

    def toggle_query_results(self):
        self.show_query_results = not self.show_query_results

    def load_file(self, filename: str):
        file_path = JUDGE_OUTPUTS_DIR / filename
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.selected_file = filename
            self.file_content = json.dumps(data, indent=2)
            self.query = data.get("query", "")
            self.answer = data.get("answer", "")
            self.docs_used = data.get("docs_used", [])
            self.judge_scores = data.get("judge_scores", {"Relevance": 0, "Groundedness": 0, "Balance": 0})

            self.human_scores = {metric: self.judge_scores.get(metric, 3)
                                 for metric in ["Relevance", "Groundedness", "Balance"]}

            self.feedback_text = ""
            self.saved_message = f"Loaded: {filename}"
        else:
            self.file_content = ""
            self.saved_message = "File not found."

    def save_feedback(self):
        if not self.selected_file:
            self.saved_message = "Please select a file first."
            return

        feedback_data = {
            "file": self.selected_file,
            "query": self.query,
            "answer": self.answer,
            "docs_used": self.docs_used,
            "judge_scores": self.judge_scores,
            "human_scores": self.human_scores,
            "text_feedback": self.feedback_text.strip() or "(No feedback provided)"
        }

        feedback_file = FEEDBACK_STORAGE_DIR / f"{self.selected_file}_human_feedback.json"
        with open(feedback_file, "w", encoding="utf-8") as f:
            json.dump(feedback_data, f, indent=2)

        self.saved_message = f"Feedback saved for {self.selected_file}"

    def update_human_score(self, metric: str, value: int):
        self.human_scores[metric] = value

    def set_feedback_text(self, text: str):
        self.feedback_text = text


def feedback_page() -> rx.Component:
    metrics = ["Relevance", "Groundedness", "Balance"]

    return rx.container(
        rx.vstack(
            rx.heading("LLM Judge Human Feedback Dashboard", size="7"),
            rx.text("Review and provide feedback for each LLM-as-a-Judge output file.", size="4"),
            rx.divider(),

            # File selection
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

            rx.divider(),

            # Query section
            rx.box(
                rx.text("Query:", size="4", weight="bold"),
                rx.text(FeedbackState.query, size="3"),
                padding_y="0.5em",
            ),

            # Query Results (RAG) collapsible
            rx.box(
                rx.button(
                    FeedbackState.query_results_button_label,
                    on_click=FeedbackState.toggle_query_results,
                    size="3",
                    color_scheme="blue",
                    margin_bottom="0.5em",
                ),
                rx.cond(
                    FeedbackState.show_query_results,
                    rx.box(
                        rx.foreach(
                            FeedbackState.docs_used,
                            lambda doc: rx.box(
                                rx.text(doc, size="3"),
                                padding="0.5em",
                                margin_y="0.25em",
                                border=rx.cond(
                                    FeedbackState.docs_used.contains(doc),
                                    "2px solid green",
                                    "1px solid #ccc"
                                ),
                                border_radius="5px",
                                white_space="pre-wrap",
                            ),
                        ),
                        padding="1em",
                        border="1px solid gray",
                        border_radius="5px",
                        overflow_y="scroll",
                        max_height="250px",
                    ),
                ),
                padding_y="0.5em",
            ),

            # LLM Analysis (Answer) collapsible
            rx.box(
                rx.button(
                    FeedbackState.analysis_button_label,
                    on_click=FeedbackState.toggle_llm_analysis,
                    size="3",
                    color_scheme="blue",
                    margin_bottom="0.5em",
                ),
                rx.cond(
                    FeedbackState.show_llm_analysis,
                    rx.box(
                        rx.text(FeedbackState.answer, size="3"),
                        padding="1em",
                        border="1px solid gray",
                        border_radius="5px",
                        overflow_y="scroll",
                        max_height="250px",
                        white_space="pre-wrap",
                    ),
                ),
                padding_y="0.5em",
            ),

            rx.divider(),

            # Judge scores
            rx.box(
                rx.text("Judge Scores:", size="4", weight="bold"),
                rx.foreach(
                    FeedbackState.judge_scores.items(),
                    lambda item: rx.text(f"{item[0]}: {item[1]}", size="3")
                ),
                padding_y="0.5em",
            ),

            rx.divider(),

            # Human score sliders
            rx.box(
                rx.text("Rate the following metrics on LLM Analysis:", size="4", weight="bold"),
                *[
                    rx.vstack(
                        rx.text(f"{metric}: {FeedbackState.human_score_display[metric]}", size="3"),
                        rx.slider(
                            min=0,
                            max=5,
                            step=1,
                            value=[FeedbackState.human_scores.get(metric, 3)],
                            on_change=lambda v, m=metric: FeedbackState.update_human_score(m, v[0]),
                        ),
                        padding_y="0.5em",
                    )
                    for metric in metrics
                ],
                padding_y="1em",
            ),

            # Textual feedback
            rx.box(
                rx.text("Your Feedback:", size="4", weight="bold"),
                rx.text_area(
                    placeholder="Enter your feedback or corrections here...",
                    value=FeedbackState.feedback_text,
                    on_change=FeedbackState.set_feedback_text,
                    min_height="150px",
                    width="100%",
                    resize="vertical",
                ),
                padding_y="1em",
            ),

            # Save button
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


# Reflex app entry
app = rx.App()
app.add_page(feedback_page, route="/", title="Human Feedback | LLM Judge Review")
