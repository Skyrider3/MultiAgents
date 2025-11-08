"""
PDF Parser with Mathematical Formula Extraction
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import fitz  # PyMuPDF
import pdfplumber
from dataclasses import dataclass
from loguru import logger

from src.config import settings, get_ingestion_logger


@dataclass
class ExtractedContent:
    """Container for extracted PDF content"""
    text: str
    formulas: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    figures: List[Dict[str, Any]]
    sections: Dict[str, str]
    metadata: Dict[str, Any]
    references: List[str]


class PDFParser:
    """
    Advanced PDF parser for mathematical documents
    """

    def __init__(self):
        self.logger = get_ingestion_logger()
        self.formula_patterns = self._compile_formula_patterns()

    def parse_pdf(self, pdf_path: Path) -> ExtractedContent:
        """
        Parse PDF and extract all content

        Args:
            pdf_path: Path to PDF file

        Returns:
            ExtractedContent object with extracted data
        """
        self.logger.info(f"Parsing PDF: {pdf_path}")

        # Extract with PyMuPDF (better for text and structure)
        text, sections, metadata = self._extract_with_pymupdf(pdf_path)

        # Extract with pdfplumber (better for tables)
        tables = self._extract_tables_with_pdfplumber(pdf_path)

        # Extract formulas from text
        formulas = self._extract_formulas(text)

        # Extract figures
        figures = self._extract_figures(pdf_path)

        # Extract references
        references = self._extract_references(text)

        content = ExtractedContent(
            text=text,
            formulas=formulas,
            tables=tables,
            figures=figures,
            sections=sections,
            metadata=metadata,
            references=references
        )

        self.logger.info(f"Extracted: {len(formulas)} formulas, {len(tables)} tables, "
                        f"{len(figures)} figures, {len(references)} references")

        return content

    def _extract_with_pymupdf(self, pdf_path: Path) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        """Extract text and structure using PyMuPDF"""
        doc = fitz.open(pdf_path)

        full_text = []
        sections = {}
        current_section = None
        section_text = []

        # Extract metadata
        metadata = {
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'subject': doc.metadata.get('subject', ''),
            'keywords': doc.metadata.get('keywords', ''),
            'pages': doc.page_count,
            'format': doc.metadata.get('format', 'PDF')
        }

        for page_num, page in enumerate(doc):
            # Extract text with structure
            blocks = page.get_text("dict")

            for block in blocks.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()

                            if text:
                                # Detect section headers
                                if self._is_section_header(text, span):
                                    # Save previous section
                                    if current_section and section_text:
                                        sections[current_section] = '\n'.join(section_text)
                                        section_text = []

                                    current_section = text
                                else:
                                    section_text.append(text)

                                full_text.append(text)

        # Save last section
        if current_section and section_text:
            sections[current_section] = '\n'.join(section_text)

        doc.close()

        return '\n'.join(full_text), sections, metadata

    def _extract_tables_with_pdfplumber(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract tables using pdfplumber"""
        tables = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()

                    for table_idx, table in enumerate(page_tables or []):
                        if table and len(table) > 1:  # At least header + 1 row
                            tables.append({
                                'page': page_num + 1,
                                'table_index': table_idx,
                                'headers': table[0] if table else [],
                                'rows': table[1:] if len(table) > 1 else [],
                                'num_rows': len(table) - 1,
                                'num_cols': len(table[0]) if table else 0
                            })

        except Exception as e:
            self.logger.warning(f"Error extracting tables: {e}")

        return tables

    def _extract_formulas(self, text: str) -> List[Dict[str, Any]]:
        """Extract mathematical formulas from text"""
        formulas = []

        # Extract display equations (between $$ or \[ \])
        display_patterns = [
            (r'\$\$(.*?)\$\$', 'latex_display'),
            (r'\\\[(.*?)\\\]', 'latex_display'),
            (r'\\begin\{equation\}(.*?)\\end\{equation\}', 'latex_equation'),
            (r'\\begin\{align\}(.*?)\\end\{align\}', 'latex_align'),
        ]

        for pattern, formula_type in display_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                formula_text = match.group(1).strip()
                if formula_text:
                    formulas.append({
                        'type': formula_type,
                        'latex': formula_text,
                        'context': text[max(0, match.start()-50):min(len(text), match.end()+50)],
                        'position': match.start()
                    })

        # Extract inline equations (between $ $)
        inline_pattern = r'\$([^\$]+)\$'
        inline_matches = re.finditer(inline_pattern, text)
        for match in inline_matches:
            formula_text = match.group(1).strip()
            # Filter out non-formula uses of $
            if self._is_likely_formula(formula_text):
                formulas.append({
                    'type': 'latex_inline',
                    'latex': formula_text,
                    'context': text[max(0, match.start()-30):min(len(text), match.end()+30)],
                    'position': match.start()
                })

        # Extract numbered equations
        numbered_pattern = r'\((\d+\.?\d*)\)\s*([^\n]+)'
        numbered_matches = re.finditer(numbered_pattern, text)
        for match in numbered_matches:
            equation_number = match.group(1)
            equation_text = match.group(2).strip()
            if self._is_likely_formula(equation_text):
                formulas.append({
                    'type': 'numbered',
                    'number': equation_number,
                    'text': equation_text,
                    'position': match.start()
                })

        # Sort by position
        formulas.sort(key=lambda x: x.get('position', 0))

        return formulas

    def _extract_figures(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract figure information from PDF"""
        figures = []
        doc = fitz.open(pdf_path)

        for page_num, page in enumerate(doc):
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                xref = img[0]
                image_info = doc.extract_image(xref)

                if image_info:
                    figures.append({
                        'page': page_num + 1,
                        'index': img_index,
                        'width': image_info.get('width'),
                        'height': image_info.get('height'),
                        'colorspace': image_info.get('colorspace'),
                        'ext': image_info.get('ext'),
                        'size': len(image_info.get('image', b''))
                    })

        doc.close()
        return figures

    def _extract_references(self, text: str) -> List[str]:
        """Extract references/bibliography from text"""
        references = []

        # Find references section
        ref_section_pattern = r'(References|Bibliography|REFERENCES|BIBLIOGRAPHY)\s*\n(.*?)(?=\n\n|\Z)'
        ref_match = re.search(ref_section_pattern, text, re.DOTALL)

        if ref_match:
            ref_text = ref_match.group(2)

            # Extract individual references (common patterns)
            ref_patterns = [
                r'\[(\d+)\]\s+([^\n]+(?:\n(?!\[\d+\])[^\n]+)*)',  # [1] Author, Title...
                r'(\d+)\.\s+([^\n]+(?:\n(?!\d+\.)[^\n]+)*)',      # 1. Author, Title...
            ]

            for pattern in ref_patterns:
                matches = re.finditer(pattern, ref_text)
                for match in matches:
                    ref_text = match.group(2).strip()
                    if ref_text:
                        references.append(ref_text)

        # Also look for arXiv references
        arxiv_pattern = r'arXiv:(\d{4}\.\d{4,5})'
        arxiv_matches = re.finditer(arxiv_pattern, text)
        for match in arxiv_matches:
            references.append(f"arXiv:{match.group(1)}")

        return references

    def _is_section_header(self, text: str, span: Dict) -> bool:
        """Check if text is likely a section header"""
        # Check font size (headers typically larger)
        font_size = span.get('size', 0)
        if font_size > 14:  # Larger font
            return True

        # Check for section patterns
        section_patterns = [
            r'^\d+\.?\s+[A-Z]',  # 1. Introduction
            r'^[IVX]+\.?\s+',     # I. Introduction
            r'^(Abstract|Introduction|Conclusion|References|Appendix)',
            r'^(Theorem|Lemma|Proposition|Corollary|Definition|Proof)\s+\d',
        ]

        for pattern in section_patterns:
            if re.match(pattern, text):
                return True

        return False

    def _is_likely_formula(self, text: str) -> bool:
        """Check if text is likely a mathematical formula"""
        # Mathematical symbols and patterns
        math_indicators = [
            r'[=<>≤≥≠∈∉⊂⊃∪∩]',
            r'\\[a-zA-Z]+',  # LaTeX commands
            r'[a-zA-Z]_[{a-zA-Z0-9}]',  # Subscripts
            r'[a-zA-Z]\^[{a-zA-Z0-9}]',  # Superscripts
            r'\d+[a-zA-Z]',  # Variables with coefficients
            r'[∑∏∫∂∇]',  # Mathematical operators
        ]

        for pattern in math_indicators:
            if re.search(pattern, text):
                return True

        # Exclude common non-formula uses of $
        if text.startswith('$') and text[1:].replace('.', '').replace(',', '').isdigit():
            return False  # Currency

        return len(text) > 2  # Not too short

    def _compile_formula_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for formula extraction"""
        patterns = {
            'theorem': re.compile(r'Theorem\s+(\d+\.?\d*)[\s:.]+(.*?)(?=Proof|Theorem|Lemma|\n\n)', re.DOTALL),
            'lemma': re.compile(r'Lemma\s+(\d+\.?\d*)[\s:.]+(.*?)(?=Proof|Theorem|Lemma|\n\n)', re.DOTALL),
            'corollary': re.compile(r'Corollary\s+(\d+\.?\d*)[\s:.]+(.*?)(?=Proof|Theorem|Lemma|\n\n)', re.DOTALL),
            'definition': re.compile(r'Definition\s+(\d+\.?\d*)[\s:.]+(.*?)(?=Definition|Theorem|Lemma|\n\n)', re.DOTALL),
            'proposition': re.compile(r'Proposition\s+(\d+\.?\d*)[\s:.]+(.*?)(?=Proof|Theorem|Lemma|\n\n)', re.DOTALL),
        }
        return patterns

    def extract_mathematical_structures(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract theorems, lemmas, definitions, etc."""
        structures = {
            'theorems': [],
            'lemmas': [],
            'corollaries': [],
            'definitions': [],
            'propositions': []
        }

        for structure_type, pattern in self.formula_patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                number = match.group(1)
                content = match.group(2).strip()

                structure_data = {
                    'number': number,
                    'content': content,
                    'type': structure_type
                }

                if structure_type == 'theorem':
                    structures['theorems'].append(structure_data)
                elif structure_type == 'lemma':
                    structures['lemmas'].append(structure_data)
                elif structure_type == 'corollary':
                    structures['corollaries'].append(structure_data)
                elif structure_type == 'definition':
                    structures['definitions'].append(structure_data)
                elif structure_type == 'proposition':
                    structures['propositions'].append(structure_data)

        return structures

    def extract_proofs(self, text: str) -> List[Dict[str, Any]]:
        """Extract proof structures from text"""
        proofs = []

        # Pattern for proofs
        proof_pattern = re.compile(
            r'Proof[\s:.]+(.*?)(?=\n\n|QED|∎|□|\Z)',
            re.DOTALL | re.IGNORECASE
        )

        matches = proof_pattern.finditer(text)
        for match in matches:
            proof_text = match.group(1).strip()

            # Determine proof type
            proof_type = 'direct'
            if 'contradiction' in proof_text.lower():
                proof_type = 'contradiction'
            elif 'induction' in proof_text.lower():
                proof_type = 'induction'
            elif 'construction' in proof_text.lower():
                proof_type = 'construction'

            proofs.append({
                'text': proof_text,
                'type': proof_type,
                'position': match.start()
            })

        return proofs

    def save_extracted_content(self, content: ExtractedContent, output_path: Path):
        """Save extracted content to JSON file"""
        output_data = {
            'text_length': len(content.text),
            'sections': content.sections,
            'metadata': content.metadata,
            'formulas_count': len(content.formulas),
            'formulas': content.formulas[:100],  # Limit for file size
            'tables_count': len(content.tables),
            'figures_count': len(content.figures),
            'references_count': len(content.references),
            'references': content.references[:50]  # Limit for file size
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

        self.logger.info(f"Saved extracted content to: {output_path}")


# Example usage
if __name__ == "__main__":
    parser = PDFParser()

    # Example PDF path
    pdf_path = Path("example.pdf")

    if pdf_path.exists():
        # Parse PDF
        content = parser.parse_pdf(pdf_path)

        # Extract mathematical structures
        structures = parser.extract_mathematical_structures(content.text)
        print(f"Found {len(structures['theorems'])} theorems")
        print(f"Found {len(structures['definitions'])} definitions")

        # Extract proofs
        proofs = parser.extract_proofs(content.text)
        print(f"Found {len(proofs)} proofs")

        # Save results
        output_path = pdf_path.with_suffix('.extracted.json')
        parser.save_extracted_content(content, output_path)