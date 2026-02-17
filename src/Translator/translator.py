# translator/translator.py
"""
Advanced Algorithm to Python Translator
========================================

Enhanced rule-based translator:
- Variables with types
- Arrays 1D & 2D
- If / Else
- For / While
- Functions / Procedures
- Logical operators
- OCR error correction
"""

import re


class AdvancedAlgoTranslator:

    def __init__(self):
        self.indent_level = 0
        self.python_lines = []

    # ==================================================
    # Utilities
    # ==================================================

    def indent(self):
        return "    " * self.indent_level

    def add_line(self, line):
        self.python_lines.append(self.indent() + line)

    def normalize(self, text):
        """
        Normalize OCR mistakes and accents.
        """
        replacements = {
            "S1 ": "Si ",
            "s1 ": "si ",
            "TantOue": "TantQue",
            "tantoue": "tantque",
            "Al0rs": "Alors",
            "F1n": "Fin",
            "é": "e",
            "è": "e",
            "à": "a"
        }

        for k, v in replacements.items():
            text = text.replace(k, v)

        return text

    # ==================================================
    # Main Translation
    # ==================================================

    def translate(self, text):

        self.indent_level = 0
        self.python_lines = []

        text = self.normalize(text)
        lines = text.split("\n")

        for raw_line in lines:

            line = raw_line.strip()
            if not line:
                continue

            lower = line.lower()

            # --------------------------------------------
            # Ignore structure words
            # --------------------------------------------
            if lower.startswith(("algorithme", "variables", "debut", "fin algorithme")):
                continue

            # --------------------------------------------
            # Variable declaration with type
            # x : Entier
            # --------------------------------------------
            if ":" in line and "tableau" not in lower:

                var = line.split(":")[0].strip()
                var_type = line.split(":")[1].strip().lower()

                default = "None"

                if "entier" in var_type:
                    default = "0"
                elif "reel" in var_type:
                    default = "0.0"
                elif "booleen" in var_type:
                    default = "False"
                elif "chaine" in var_type:
                    default = "''"

                self.add_line(f"{var} = {default}")
                continue

            # --------------------------------------------
            # Array 1D
            # T : Tableau[10]
            # --------------------------------------------
            if "tableau" in lower and "[" in lower:

                match_1d = re.search(r"(\w+)\s*:\s*tableau\[(\d+)\]", lower)
                match_2d = re.search(r"(\w+)\s*:\s*tableau\[(\d+)\]\[(\d+)\]", lower)

                if match_2d:
                    name = match_2d.group(1)
                    r = match_2d.group(2)
                    c = match_2d.group(3)
                    self.add_line(f"{name} = [[0]*{c} for _ in range({r})]")
                elif match_1d:
                    name = match_1d.group(1)
                    size = match_1d.group(2)
                    self.add_line(f"{name} = [0]*{size}")
                continue

            # --------------------------------------------
            # Assignment
            # --------------------------------------------
            if "<-" in line:
                left, right = line.split("<-")
                right = self.convert_condition(right)
                self.add_line(f"{left.strip()} = {right.strip()}")
                continue

            # --------------------------------------------
            # Input
            # --------------------------------------------
            if lower.startswith("lire"):
                var = re.findall(r"\((.*?)\)", line)
                if var:
                    self.add_line(f"{var[0]} = input()")
                continue

            # --------------------------------------------
            # Output
            # --------------------------------------------
            if lower.startswith("ecrire"):
                var = re.findall(r"\((.*?)\)", line)
                if var:
                    self.add_line(f"print({var[0]})")
                continue

            # --------------------------------------------
            # IF
            # --------------------------------------------
            if lower.startswith("si"):
                condition = lower.replace("si", "")
                condition = condition.replace("alors", "")
                condition = self.convert_condition(condition)
                self.add_line(f"if {condition.strip()}:")
                self.indent_level += 1
                continue

            if lower.startswith("sinon"):
                self.indent_level -= 1
                self.add_line("else:")
                self.indent_level += 1
                continue

            if lower.startswith("fin si"):
                self.indent_level -= 1
                continue

            # --------------------------------------------
            # FOR
            # --------------------------------------------
            if lower.startswith("pour"):
                match = re.search(r"pour\s+(\w+)\s+de\s+(\d+)\s+a\s+(\d+)", lower)
                if match:
                    var = match.group(1)
                    start = match.group(2)
                    end = match.group(3)
                    self.add_line(f"for {var} in range({start}, {int(end)+1}):")
                    self.indent_level += 1
                continue

            if lower.startswith("fin pour"):
                self.indent_level -= 1
                continue

            # --------------------------------------------
            # WHILE
            # --------------------------------------------
            if lower.startswith("tantque"):
                condition = lower.replace("tantque", "")
                condition = condition.replace("faire", "")
                condition = self.convert_condition(condition)
                self.add_line(f"while {condition.strip()}:")
                self.indent_level += 1
                continue

            if lower.startswith("fin tantque"):
                self.indent_level -= 1
                continue

            # --------------------------------------------
            # FUNCTION
            # --------------------------------------------
            if lower.startswith("fonction"):
                match = re.search(r"fonction\s+(\w+)\((.*?)\)", lower)
                if match:
                    name = match.group(1)
                    params = match.group(2)
                    self.add_line(f"def {name}({params}):")
                    self.indent_level += 1
                continue

            # --------------------------------------------
            # PROCEDURE
            # --------------------------------------------
            if lower.startswith("procedure"):
                match = re.search(r"procedure\s+(\w+)\((.*?)\)", lower)
                if match:
                    name = match.group(1)
                    params = match.group(2)
                    self.add_line(f"def {name}({params}):")
                    self.indent_level += 1
                continue

            if lower.startswith(("fin fonction", "fin procedure")):
                self.indent_level -= 1
                continue

            # --------------------------------------------
            # RETURN
            # --------------------------------------------
            if lower.startswith("retourner"):
                value = lower.replace("retourner", "")
                self.add_line(f"return {value.strip()}")
                continue

        return "\n".join(self.python_lines)

    # ==================================================
    # Condition Conversion
    # ==================================================

    def convert_condition(self, condition):

        condition = condition.replace("<>", "!=")
        condition = condition.replace("=", "==")
        condition = condition.replace("====", "==")
        condition = condition.replace("<==", "<=")
        condition = condition.replace(">==", ">=")

        condition = condition.replace(" et ", " and ")
        condition = condition.replace(" ou ", " or ")
        condition = condition.replace(" non ", " not ")

        return condition.strip()


# ==================================================
# Public function
# ==================================================

def translate_to_python(text):
    translator = AdvancedAlgoTranslator()
    return translator.translate(text)
