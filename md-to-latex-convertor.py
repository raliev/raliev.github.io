#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
import os
import argparse

class MarkdownToLatexConverter:
    """
    A self-contained converter for a specific Markdown format to a LaTeX book,
    with an integrated template.
    """
    def __init__(self, md_path, output_path, paper_size='a4paper'):
        self.md_path = md_path
        self.output_path = output_path
        self.paper_size = paper_size
        self.front_matter = {}
        self.latex_body = ""
        self.abstract_content = ""

    def _get_latex_template(self):
        """Returns the hardcoded LaTeX book template."""
        return f"""
\\documentclass[{self.paper_size}, 11pt, captions=tableheading]{{scrbook}}

% Packages
\\usepackage[utf8]{{inputenc}}
\\usepackage{{xcolor}}
\\usepackage[margin=2.5cm, includehead=true, includefoot=true, centering]{{geometry}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{float}} % For [H] figure placement
\\usepackage[hang,flushmargin,bottom,multiple]{{footmisc}}
\\usepackage{{hyperref}}
\\usepackage{{microtype}} % Improves typography and helps avoid overfull hboxes

% --- FIX: Define abstractname and the abstract environment for scrbook ---
\\newcommand{{\\abstractname}}{{Abstract}}
\\newenvironment{{abstract}}
  {{\\begin{{center}}\\bfseries\\abstractname\\end{{center}}\\begin{{quotation}}}}
  {{\\end{{quotation}}}}

% --- FIX: Define graphics path to look for images in parent directories ---
\\graphicspath{{{{./}}{{../}}{{../../}}}}

% --- FIX: Remove paragraph indentation and add space between paragraphs ---
\\setlength{{\\parindent}}{{0pt}}
\\setlength{{\\parskip}}{{1em}}

% --- FIX: Set numbering depth for sections and table of contents ---
\\setcounter{{secnumdepth}}{{5}}
\\setcounter{{tocdepth}}{{5}}

% Hyperref setup
\\hypersetup{{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,
    urlcolor=cyan,
    pdftitle={{[TITLE]}},
    pdfpagemode=FullScreen,
}}

% Document metadata placeholders
\\title{{[TITLE]}}
\\author{{[AUTHOR]}}
\\date{{[DATE]}}

\\begin{{document}}

\\frontmatter
\\maketitle

[TOC_MARKER]

[ABSTRACT_MARKER]

\\mainmatter

[BODY_MARKER]

\\end{{document}}
"""

    def run(self):
        """Executes the full conversion process."""
        print("Starting conversion...")
        md_content = self._read_file(self.md_path)
        template_content = self._get_latex_template()

        self._parse_markdown(md_content)
        final_latex = self._apply_template(template_content)

        self._write_file(self.output_path, final_latex)
        print(f"Conversion successful. Output written to '{self.output_path}'")

    def _read_file(self, path):
        """Reads a file and returns its content."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: File not found at '{path}'")
            sys.exit(1)

    def _write_file(self, path, content):
        """Writes content to a file."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

    def _escape_latex(self, text):
        """Escapes special LaTeX characters."""
        conv = {
            '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
            '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}', '\\': r'\textbackslash{}',
        }
        regex = re.compile('|'.join(re.escape(key) for key in sorted(conv.keys(), key=lambda item: - len(item))))
        return regex.sub(lambda match: conv[match.group()], text)

    def _process_inline_markdown(self, text):
        """Processes inline markdown like bold, italics, links, and citations."""
        # Pre-process and remove problematic unicode characters
        text = text.replace('≈', '$\\approx$') # Replace with LaTeX math command
        text = text.replace('🐢', '') # Remove emoji
        text = text.replace('💬', '') # Remove emoji

        # Protect math expressions from escaping
        math_expressions = []
        def store_math(match):
            math_expressions.append(match.group(0))
            return f"LATEXMATHBLOCK{len(math_expressions)-1}"

        # Use a more specific regex for inline math (don't span newlines)
        text = re.sub(r'\$(.*?)\$', store_math, text)

        # Escape the rest of the text
        text = self._escape_latex(text)

        # Markdown to LaTeX conversions
        text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', text)
        text = re.sub(r'\*(.*?)\*', r'\\textit{\1}', text)
        text = re.sub(r'`(.*?)`', r'\\texttt{\1}', text)
        text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\\href{\2}{\1}', text)
        text = re.sub(r'<a id="(.+?)"></a>', r'\\label{\1}', text)

        # Restore math expressions
        for i, math_expr in enumerate(math_expressions):
            text = text.replace(f"LATEXMATHBLOCK{i}", math_expr)

        return text

    def _parse_markdown(self, md_content):
        """Parses the entire markdown content into its components."""
        parts = md_content.split('---', 2)
        if len(parts) < 3:
            raise ValueError("Markdown file does not contain a valid YAML front matter block.")

        # Parse front matter with support for multi-line lists
        yaml_lines = parts[1].strip().split('\n')
        self.front_matter = {}
        current_key = None
        for line in yaml_lines:
            match_kv = re.match(r'^([a-zA-Z_]+):\s*(.*)', line)
            if match_kv:
                current_key = match_kv.group(1)
                value = match_kv.group(2).strip().strip('"')
                if value:
                    self.front_matter[current_key] = value
                else:
                    self.front_matter[current_key] = []
            elif (match_li := re.match(r'^\s*-\s+(.*)', line)) and current_key and isinstance(self.front_matter.get(current_key), list):
                self.front_matter[current_key].append(match_li.group(1).strip())

        body_str = parts[2]

        # Pre-process and store display math blocks
        display_math_blocks = []
        def store_display_math(match):
            display_math_blocks.append(match.group(1).strip())
            return f"\nLATEXMATHDISPLAYBLOCK{len(display_math_blocks)-1}\n"
        body_str = re.sub(r'\s*\$\$(.*?)\$\$\s*', store_display_math, body_str, flags=re.DOTALL)

        # Parse body line by line
        latex_lines, abstract_lines = [], []
        lines = body_str.strip().split('\n')
        in_itemize, in_enumerate, in_quote, is_abstract = False, False, False, False
        paragraph_buffer = []

        def flush_paragraph():
            nonlocal paragraph_buffer
            if paragraph_buffer:
                # Add vertical space if the paragraph starts with bold text
                if paragraph_buffer[0].strip().startswith('**'):
                    latex_lines.append('\\smallskip')

                full_para = ' '.join(paragraph_buffer).strip()
                if full_para:
                    latex_lines.append(self._process_inline_markdown(full_para))
                    # --- FIX: Add an empty line to create a paragraph break in LaTeX ---
                    latex_lines.append('')
                paragraph_buffer = []

        for line in lines:
            line = line.rstrip()

            if not re.match(r'^\s*[\*\-]\s+', line) and in_itemize:
                latex_lines.append('\\end{itemize}\n'); in_itemize = False
            if not re.match(r'^\s*\d+\.\s+', line) and in_enumerate:
                latex_lines.append('\\end{enumerate}\n'); in_enumerate = False
            if not re.match(r'^>\s*', line) and in_quote:
                latex_lines.append('\\end{quote}\n'); in_quote = False

            if not line.strip():
                flush_paragraph(); continue

            if re.match(r'^##\s+Abstract', line, re.IGNORECASE):
                flush_paragraph(); is_abstract = True; continue

            if is_abstract:
                if re.match(r'^##', line): is_abstract = False
                else: abstract_lines.append(line); continue

            # Corrected heading hierarchy for book class
            if m := re.match(r'^(#####)\d+\.\s*(.*)', line): # Numbered subsubsection
                flush_paragraph(); latex_lines.append(f'\\subsubsection{{{self._process_inline_markdown(m.group(2))}}}\n')
            elif m := re.match(r'^(#####)\s*(.*)', line): # Unnumbered subsubsection
                flush_paragraph(); latex_lines.append(f'\\subsubsection*{{{self._process_inline_markdown(m.group(2))}}}\n')
            elif m := re.match(r'^(####)\d+\.\s*(.*)', line): # Numbered subsection
                flush_paragraph(); latex_lines.append(f'\\subsection{{{self._process_inline_markdown(m.group(2))}}}\n')
            elif m := re.match(r'^(####)\s*(.*)', line): # Unnumbered subsection
                flush_paragraph(); latex_lines.append(f'\\subsection*{{{self._process_inline_markdown(m.group(2))}}}\n')
            elif m := re.match(r'^(###)\d+\.\s*(.*)', line): # Numbered section
                flush_paragraph(); latex_lines.append(f'\\section{{{self._process_inline_markdown(m.group(2))}}}\n')
            elif m := re.match(r'^(###)\s*(.*)', line): # Unnumbered section
                flush_paragraph(); latex_lines.append(f'\\section*{{{self._process_inline_markdown(m.group(2))}}}\n')
            elif m := re.match(r'^(##)\d+\.\s*(.*)', line): # Numbered chapter
                flush_paragraph(); latex_lines.append(f'\\chapter{{{self._process_inline_markdown(m.group(2))}}}\n')
            elif m := re.match(r'^(##)\s*(.*)', line): # Unnumbered chapter
                flush_paragraph(); latex_lines.append(f'\\chapter*{{{self._process_inline_markdown(m.group(2))}}}\n')
            elif re.match(r'^\s*---\s*$', line):
                flush_paragraph(); latex_lines.append('\\smallskip\n')
            elif m := re.match(r'^\s*!\[(.*?)\]\((.*?)\)\s*$', line):
                flush_paragraph()
                img_path = m.group(2).lstrip('/')
                latex_lines.extend(['\\begin{figure}[H]', '  \\centering', f'  \\includegraphics[width=0.8\\textwidth]{{{img_path}}}', '\\end{figure}\n'])
            elif m := re.match(r'^\s*[\*\-]\s+(.*)', line):
                flush_paragraph()
                if not in_itemize: latex_lines.append('\\begin{itemize}'); in_itemize = True
                latex_lines.append(f'  \\item {self._process_inline_markdown(m.group(1))}')
            elif m := re.match(r'^\s*\d+\.\s+(.*)', line):
                flush_paragraph()
                if not in_enumerate: latex_lines.append('\\begin{enumerate}'); in_enumerate = True
                latex_lines.append(f'  \\item {self._process_inline_markdown(m.group(1))}')
            elif m := re.match(r'^>\s*(.*)', line):
                flush_paragraph()
                if not in_quote: latex_lines.append('\\begin{quote}'); in_quote = True
                latex_lines.append(self._process_inline_markdown(m.group(1)))
            elif re.match(r'^LATEXMATHDISPLAYBLOCK\d+$', line.strip()):
                flush_paragraph(); latex_lines.append(line.strip())
            else:
                paragraph_buffer.append(line)

        flush_paragraph()
        if in_itemize: latex_lines.append('\\end{itemize}\n')
        if in_enumerate: latex_lines.append('\\end{enumerate}\n')
        if in_quote: latex_lines.append('\\end{quote}\n')

        self.abstract_content = ' '.join(abstract_lines)
        body_with_placeholders = '\n'.join(latex_lines)

        for i, math_content in enumerate(display_math_blocks):
            placeholder = f"LATEXMATHDISPLAYBLOCK{i}"
            latex_math = f"\\begin{{equation*}}\n    {math_content}\n\\end{{equation*}}"
            body_with_placeholders = body_with_placeholders.replace(placeholder, latex_math)

        self.latex_body = body_with_placeholders

    def _apply_template(self, template_content):
        """Fills the hardcoded LaTeX template with the parsed content."""
        final_latex = template_content

        # Replace simple placeholders
        final_latex = final_latex.replace('[TITLE]', self.front_matter.get('title', ''))
        final_latex = final_latex.replace('[AUTHOR]', self.front_matter.get('author', ''))
        final_latex = final_latex.replace('[DATE]', self.front_matter.get('date', ''))

        # Handle TOC
        if str(self.front_matter.get('toc', 'false')).lower() == 'true':
            final_latex = final_latex.replace('[TOC_MARKER]', '\\tableofcontents\n\\newpage')
        else:
            final_latex = final_latex.replace('[TOC_MARKER]', '')

        # Handle Abstract
        if self.abstract_content:
            processed_abstract = self._process_inline_markdown(self.abstract_content)
            abstract_block = f'\\begin{{abstract}}\n{processed_abstract}\n\\end{{abstract}}\n\\newpage'
            final_latex = final_latex.replace('[ABSTRACT_MARKER]', abstract_block)
        else:
            final_latex = final_latex.replace('[ABSTRACT_MARKER]', '')

        # Replace the main body
        final_latex = final_latex.replace('[BODY_MARKER]', self.latex_body)

        return final_latex

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a Markdown file to a LaTeX book.")
    parser.add_argument("markdown_file", help="Path to the input Markdown file.")
    parser.add_argument("-o", "--output", help="Path to the output .tex file (optional).")
    parser.add_argument("-p", "--paper", choices=['a4paper', 'letterpaper'], default='a4paper', help="Paper size (a4paper or letterpaper). Default: a4paper")

    args = parser.parse_args()

    md_file = args.markdown_file
    paper_size = args.paper

    if args.output:
        output_file = args.output
    else:
        output_dir = os.path.dirname(os.path.abspath(md_file))
        output_filename = os.path.splitext(os.path.basename(md_file))[0] + ".tex"
        output_file = os.path.join(output_dir, output_filename)

    converter = MarkdownToLatexConverter(
        md_path=md_file,
        output_path=output_file,
        paper_size=paper_size
    )
    converter.run()

