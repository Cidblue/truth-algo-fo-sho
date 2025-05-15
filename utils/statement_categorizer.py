"""
Statement Categorizer - GUI tool for manually categorizing statements
according to the 28 outpoints and pluspoints defined in the Truth Algorithm.

Usage:
    python -m utils.statement_categorizer input.txt output.csv
"""

import sys
import csv
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import nltk
from pathlib import Path
from typing import List, Dict, Set, Optional
import re


def ensure_nltk_data():
    """Ensure NLTK data is downloaded, with fallback options."""
    try:
        # Download both punkt and punkt_tab
        nltk.download('punkt', quiet=True)

        # Try to directly download punkt_tab if that's what's missing
        try:
            nltk.download('punkt_tab', quiet=True)
        except:
            pass

        # Try a more direct approach to download all punkt data
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        nltk.download('punkt', quiet=True)

        # Verify it works by importing and using it
        from nltk.tokenize import sent_tokenize
        test = sent_tokenize("This is a test. This is another test.")
        if len(test) == 2:
            print("NLTK punkt tokenizer successfully installed and working.")
            return True
    except Exception as e:
        print(f"Failed to set up NLTK punkt tokenizer: {e}")
        print("Will use regex-based sentence splitting as fallback.")
        return False


# Try to set up NLTK data
ensure_nltk_data()

# Define the correct outpoints and pluspoints from Investigations.txt
# These are the official 14 outpoints and 14 pluspoints
outpoint_descriptions = {
    "omitted_data": "An omitted anything is an outpoint. This can be an omitted person, terminal, object, energy, space, time, form, sequence, or even an omitted scene.",
    "altered_sequence": "Any things, events, objects, sizes in a wrong sequence is an outpoint.",
    "dropped_time": "Time that should be noted and isn't would be an outpoint of dropped time. It is a special case of an omitted datum.",
    "falsehood": "When you hear two facts that are contrary, one is a falsehood or both are. A false anything qualifies for this outpoint.",
    "altered_importance": "An importance shifted from its actual relative importance, up or down. An outpoint.",
    "wrong_target": "Mistaken objective wherein one believes he is or should be reaching toward A and finds he is or should be reaching toward B is an outpoint.",
    "wrong_source": "Information taken from wrong source, orders taken from the wrong source, gifts or materiel taken from wrong source all add up to eventual confusion and possible trouble.",
    "contrary_facts": "When two statements are made on one subject which are contrary to each other, we have contrary facts.",
    "added_time": "In this outpoint we have the reverse of dropped time. In added time we have, as the most common example, something taking longer than it possibly could.",
    "added_inapplicable_data": "Just plain added data does not necessarily constitute an outpoint. It may be someone being thorough. But when the data is in no way applicable to the scene or situation and is added it is a definite outpoint.",
    "incorrectly_included_datum": "A part from one class of parts is included wrongly in another class of parts. So there is an incorrectly included datum which is a companion to the omitted datum as an outpoint.",
    "assumed_identities_not_identical": "This outpoint occurs when things that are actually different are treated as if they're identical.",
    "assumed_similarities_not_similar": "This outpoint occurs when things that don't share meaningful characteristics are treated as if they're similar.",
    "assumed_differences_not_different": "This outpoint occurs when things that are actually identical or the same class are treated as if they're different."
}

pluspoint_descriptions = {
    "related_facts_known": "All relevant facts known.",
    "events_in_correct_sequence": "Events in actual sequence.",
    "time_noted": "Time is properly noted.",
    "data_proven_factual": "Data must be factual, which is to say, true and valid.",
    "correct_relative_importance": "The important and unimportant are correctly sorted out.",
    "expected_time_period": "Events occurring or done in the time one would reasonably expect them to be.",
    "adequate_data": "No sectors of omitted data that would influence the situation.",
    "applicable_data": "The data presented or available applies to the matter in hand and not something else.",
    "correct_source": "Not wrong source.",
    "correct_target": "Not going in some direction that would be wrong for the situation.",
    "data_in_same_classification": "Data from two or more different classes of material not introduced as the same class.",
    "identities_are_identical": "Not similar or different.",
    "similarities_are_similar": "Things that share meaningful characteristics are recognized as similar.",
    "differences_are_different": "Things that are actually different are recognized as different."
}

# Try to import from rules.py, but use our definitions if that fails
sys.path.append(str(Path(__file__).parent.parent))
try:
    from rules.rules import outpoint_descriptions as imported_outpoints
    from rules.rules import pluspoint_descriptions as imported_pluspoints

    # Verify we have exactly 14 outpoints and 14 pluspoints
    if len(imported_outpoints) == 14 and len(imported_pluspoints) == 14:
        outpoint_descriptions = imported_outpoints
        pluspoint_descriptions = imported_pluspoints
    else:
        print("Warning: Imported outpoints/pluspoints don't have exactly 14 items each. Using built-in definitions.")
except ImportError:
    print("Using built-in outpoint and pluspoint definitions.")


class StatementCategorizerApp:
    def __init__(self, root, input_text="", output_file=None):
        self.root = root
        self.root.title("Truth Algorithm Statement Categorizer")
        self.root.geometry("1200x800")

        self.statements = []
        self.current_index = 0
        self.statement_labels: Dict[int, Set[str]] = {}
        self.output_file = output_file

        # Create categories from outpoints and pluspoints
        self.categories = {
            "Outpoints": list(outpoint_descriptions.keys()),
            "Pluspoints": list(pluspoint_descriptions.keys())
        }

        # Verify we have exactly 14 outpoints and 14 pluspoints
        if len(self.categories["Outpoints"]) != 14 or len(self.categories["Pluspoints"]) != 14:
            print(
                f"Warning: Expected 14 outpoints and 14 pluspoints, but got {len(self.categories['Outpoints'])} outpoints and {len(self.categories['Pluspoints'])} pluspoints.")

        self.create_widgets()

        if input_text:
            self.load_text(input_text)

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top section - Text input and file controls
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(top_frame, text="Load Text",
                   command=self.load_text_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Set Output File",
                   command=self.set_output_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Save Progress",
                   command=self.save_progress).pack(side=tk.LEFT, padx=5)

        # Middle section - Current statement display
        statement_frame = ttk.LabelFrame(
            main_frame, text="Current Statement", padding="10")
        statement_frame.pack(fill=tk.BOTH, expand=False, pady=10)

        self.statement_text = scrolledtext.ScrolledText(
            statement_frame, wrap=tk.WORD, height=5)
        self.statement_text.pack(fill=tk.BOTH, expand=True)
        self.statement_text.config(state=tk.DISABLED)

        # Navigation controls
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=5)

        ttk.Button(nav_frame, text="Previous",
                   command=self.prev_statement).pack(side=tk.LEFT, padx=5)
        self.status_label = ttk.Label(nav_frame, text="Statement 0 of 0")
        self.status_label.pack(side=tk.LEFT, padx=20)
        ttk.Button(nav_frame, text="Next", command=self.next_statement).pack(
            side=tk.LEFT, padx=5)

        # Bottom section - Category selection
        categories_frame = ttk.LabelFrame(
            main_frame, text="Categories", padding="10")
        categories_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Create notebook with tabs for Outpoints, Pluspoints, and None
        self.notebook = ttk.Notebook(categories_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Outpoints tab
        outpoints_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(outpoints_frame, text="Outpoints")

        # Pluspoints tab
        pluspoints_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(pluspoints_frame, text="Pluspoints")

        # None tab (for statements that don't match any category)
        none_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(none_frame, text="None")

        ttk.Button(none_frame, text="Mark as None",
                   command=self.mark_as_none).pack(pady=20)

        # Create checkbuttons for outpoints
        self.outpoint_vars = {}
        for i, outpoint in enumerate(self.categories["Outpoints"]):
            var = tk.BooleanVar()
            self.outpoint_vars[outpoint] = var

            frame = ttk.Frame(outpoints_frame)
            frame.grid(row=i//2, column=i % 2, sticky="w", padx=10, pady=5)

            # Format the display name properly
            display_name = outpoint.replace("_", " ").title()

            cb = ttk.Checkbutton(frame, text=display_name, variable=var)
            cb.pack(side=tk.LEFT)

            # Add tooltip/help button
            help_btn = ttk.Button(frame, text="?", width=2,
                                  command=lambda op=outpoint: self.show_description("Outpoint", op))
            help_btn.pack(side=tk.LEFT, padx=5)

        # Create checkbuttons for pluspoints
        self.pluspoint_vars = {}
        for i, pluspoint in enumerate(self.categories["Pluspoints"]):
            var = tk.BooleanVar()
            self.pluspoint_vars[pluspoint] = var

            frame = ttk.Frame(pluspoints_frame)
            frame.grid(row=i//2, column=i % 2, sticky="w", padx=10, pady=5)

            # Format the display name properly
            display_name = pluspoint.replace("_", " ").title()

            cb = ttk.Checkbutton(frame, text=display_name, variable=var)
            cb.pack(side=tk.LEFT)

            # Add tooltip/help button
            help_btn = ttk.Button(frame, text="?", width=2,
                                  command=lambda pp=pluspoint: self.show_description("Pluspoint", pp))
            help_btn.pack(side=tk.LEFT, padx=5)

    def show_description(self, category_type, category_name):
        """Show a popup with the description of the selected category"""
        if category_type == "Outpoint":
            description = outpoint_descriptions.get(
                category_name, "No description available")
        else:
            description = pluspoint_descriptions.get(
                category_name, "No description available")

        messagebox.showinfo(
            f"{category_name.replace('_', ' ').title()} Description",
            description
        )

    def load_text_file(self):
        """Open a file dialog to load text from a file"""
        file_path = filedialog.askopenfilename(
            title="Select Text File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                self.load_text(text)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")

    def load_text(self, text):
        """Process the input text and split it into statements"""
        # Try multiple approaches to tokenize the text
        nltk_success = False

        # Approach 1: Try using NLTK's sent_tokenize
        try:
            from nltk.tokenize import sent_tokenize
            self.statements = sent_tokenize(text)
            if len(self.statements) > 0:
                print(
                    f"Successfully tokenized text into {len(self.statements)} statements using NLTK sent_tokenize.")
                nltk_success = True
        except Exception as e:
            print(f"NLTK sent_tokenize failed: {e}")

        # Approach 2: Try using NLTK's PunktSentenceTokenizer directly
        if not nltk_success:
            try:
                from nltk.tokenize.punkt import PunktSentenceTokenizer
                tokenizer = PunktSentenceTokenizer()
                self.statements = tokenizer.tokenize(text)
                if len(self.statements) > 0:
                    print(
                        f"Successfully tokenized text into {len(self.statements)} statements using PunktSentenceTokenizer.")
                    nltk_success = True
            except Exception as e:
                print(f"NLTK PunktSentenceTokenizer failed: {e}")

        # Fallback: Use regex-based sentence splitting
        if not nltk_success:
            print("Using regex fallback for sentence tokenization.")
            # Split on period, question mark, or exclamation point followed by space or newline
            self.statements = re.split(r'(?<=[.!?])\s+', text)
            # Remove empty statements and strip whitespace
            self.statements = [s.strip() for s in self.statements if s.strip()]
            print(
                f"Tokenized text into {len(self.statements)} statements using regex fallback.")

        # If we only have one statement but it contains multiple lines, split by lines
        if len(self.statements) == 1 and '\n' in self.statements[0]:
            lines = self.statements[0].split('\n')
            # Filter out empty lines
            self.statements = [line.strip() for line in lines if line.strip()]
            print(f"Split single statement into {len(self.statements)} lines.")

        self.current_index = 0
        self.statement_labels = {i: set() for i in range(len(self.statements))}

        # Update the display
        self.update_display()
        messagebox.showinfo(
            "Text Loaded", f"Loaded {len(self.statements)} statements.")

    def set_output_file(self):
        """Open a file dialog to set the output CSV file"""
        file_path = filedialog.asksaveasfilename(
            title="Save Categorized Statements",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if file_path:
            self.output_file = file_path
            messagebox.showinfo(
                "Output File", f"Output will be saved to: {file_path}")

    def update_display(self):
        """Update the UI to show the current statement and its labels"""
        if not self.statements:
            return

        # Update statement text
        self.statement_text.config(state=tk.NORMAL)
        self.statement_text.delete(1.0, tk.END)
        self.statement_text.insert(tk.END, self.statements[self.current_index])
        self.statement_text.config(state=tk.DISABLED)

        # Update status label
        self.status_label.config(
            text=f"Statement {self.current_index + 1} of {len(self.statements)}")

        # Update checkboxes based on current labels
        current_labels = self.statement_labels.get(self.current_index, set())

        # Reset all checkboxes
        for var in self.outpoint_vars.values():
            var.set(False)
        for var in self.pluspoint_vars.values():
            var.set(False)

        # Set checkboxes based on current labels
        for label in current_labels:
            if label in self.outpoint_vars:
                self.outpoint_vars[label].set(True)
            elif label in self.pluspoint_vars:
                self.pluspoint_vars[label].set(True)

    def save_current_labels(self):
        """Save the current selection of labels for the current statement"""
        if not self.statements:
            return

        # Get selected outpoints
        selected_outpoints = [op for op,
                              var in self.outpoint_vars.items() if var.get()]

        # Get selected pluspoints
        selected_pluspoints = [pp for pp,
                               var in self.pluspoint_vars.items() if var.get()]

        # Combine all selected labels
        selected_labels = set(selected_outpoints + selected_pluspoints)

        # Save to statement_labels dictionary
        self.statement_labels[self.current_index] = selected_labels

    def mark_as_none(self):
        """Mark the current statement as having no categories"""
        if not self.statements:
            return

        # Clear all checkboxes
        for var in self.outpoint_vars.values():
            var.set(False)
        for var in self.pluspoint_vars.values():
            var.set(False)

        # Save empty set for this statement
        self.statement_labels[self.current_index] = set()

        # Move to next statement
        self.next_statement()

    def next_statement(self):
        """Move to the next statement"""
        if not self.statements:
            return

        # Save current labels
        self.save_current_labels()

        # Move to next statement if not at the end
        if self.current_index < len(self.statements) - 1:
            self.current_index += 1
            self.update_display()
        else:
            messagebox.showinfo(
                "End Reached", "You've reached the last statement.")

    def prev_statement(self):
        """Move to the previous statement"""
        if not self.statements:
            return

        # Save current labels
        self.save_current_labels()

        # Move to previous statement if not at the beginning
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()
        else:
            messagebox.showinfo("Beginning Reached",
                                "You're at the first statement.")

    def save_progress(self):
        """Save the current progress to the output file"""
        if not self.statements:
            messagebox.showwarning("No Data", "No statements to save.")
            return

        if not self.output_file:
            self.set_output_file()
            if not self.output_file:
                return

        # Save current labels before writing to file
        self.save_current_labels()

        try:
            with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Statement', 'Category', 'Type'])

                for idx, labels in self.statement_labels.items():
                    statement = self.statements[idx]

                    if not labels:
                        # Write statement with "None" category
                        writer.writerow([statement, 'None', 'None'])
                    else:
                        # Write statement with each category
                        for label in labels:
                            category_type = 'Outpoint' if label in self.outpoint_vars else 'Pluspoint'
                            writer.writerow([statement, label, category_type])

            messagebox.showinfo(
                "Save Successful", f"Saved {len(self.statements)} statements to {self.output_file}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save: {str(e)}")


def main():
    """Main entry point for the statement categorizer"""
    input_file = None
    output_file = None

    # Parse command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    # Initialize the GUI
    root = tk.Tk()

    # Set initial text if input file provided
    initial_text = ""
    if input_file:
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                initial_text = f.read()
        except Exception as e:
            print(f"Error loading input file: {str(e)}")

    app = StatementCategorizerApp(root, initial_text, output_file)

    # Start the main loop
    root.mainloop()


if __name__ == "__main__":
    main()
