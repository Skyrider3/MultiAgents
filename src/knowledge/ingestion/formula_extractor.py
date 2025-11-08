"""
Mathematical Formula Extractor with OCR Support
Integrates multiple methods for formula extraction including DeepSeek-OCR
"""

import re
import json
import base64
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image
import io
from loguru import logger

from src.config import settings, get_ingestion_logger


@dataclass
class Formula:
    """Represents an extracted mathematical formula"""
    latex: str
    mathml: Optional[str] = None
    ascii_math: Optional[str] = None
    image: Optional[bytes] = None
    confidence: float = 0.0
    source: str = "text"  # text, ocr, image
    context: Optional[str] = None
    variables: List[str] = None
    formula_type: str = "equation"  # equation, inequality, definition, etc.


class FormulaExtractor:
    """
    Advanced formula extraction from PDFs and images
    Supports text extraction, OCR, and specialized math OCR services
    """

    def __init__(self, use_ocr: bool = True):
        self.logger = get_ingestion_logger()
        self.use_ocr = use_ocr and settings.documents.enable_formula_extraction

        # Initialize DeepSeek-OCR if available
        self.deepseek_available = self._check_deepseek_availability()

        # LaTeX command patterns
        self.latex_commands = self._compile_latex_patterns()

        # Mathematical symbol mappings
        self.symbol_map = self._create_symbol_map()

    def extract_formulas_from_text(self, text: str) -> List[Formula]:
        """
        Extract formulas from plain text using pattern matching

        Args:
            text: Input text containing formulas

        Returns:
            List of extracted Formula objects
        """
        formulas = []

        # Extract LaTeX formulas
        latex_formulas = self._extract_latex_formulas(text)
        formulas.extend(latex_formulas)

        # Extract ASCII math notation
        ascii_formulas = self._extract_ascii_formulas(text)
        formulas.extend(ascii_formulas)

        # Extract Unicode mathematical notation
        unicode_formulas = self._extract_unicode_formulas(text)
        formulas.extend(unicode_formulas)

        # Deduplicate and sort
        formulas = self._deduplicate_formulas(formulas)

        self.logger.info(f"Extracted {len(formulas)} formulas from text")
        return formulas

    def extract_formulas_from_image(self, image_path: Path) -> List[Formula]:
        """
        Extract formulas from image using OCR

        Args:
            image_path: Path to image file

        Returns:
            List of extracted Formula objects
        """
        if not self.use_ocr:
            self.logger.warning("OCR disabled, skipping image formula extraction")
            return []

        formulas = []

        # Try DeepSeek-OCR first if available
        if self.deepseek_available:
            try:
                deepseek_formulas = self._extract_with_deepseek(image_path)
                formulas.extend(deepseek_formulas)
            except Exception as e:
                self.logger.error(f"DeepSeek-OCR failed: {e}")

        # Fallback to basic OCR with preprocessing
        if not formulas:
            try:
                basic_formulas = self._extract_with_basic_ocr(image_path)
                formulas.extend(basic_formulas)
            except Exception as e:
                self.logger.error(f"Basic OCR failed: {e}")

        return formulas

    def _extract_latex_formulas(self, text: str) -> List[Formula]:
        """Extract LaTeX formulas from text"""
        formulas = []

        # Display math patterns
        display_patterns = [
            (r'\$\$(.*?)\$\$', 'display'),
            (r'\\\[(.*?)\\\]', 'display'),
            (r'\\begin\{equation\}(.*?)\\end\{equation\}', 'equation'),
            (r'\\begin\{align\}(.*?)\\end\{align\}', 'align'),
            (r'\\begin\{gather\}(.*?)\\end\{gather\}', 'gather'),
            (r'\\begin\{multline\}(.*?)\\end\{multline\}', 'multline'),
        ]

        # Inline math pattern
        inline_pattern = r'\$([^\$]+)\$'

        # Extract display formulas
        for pattern, formula_type in display_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                latex_code = match.group(1).strip()
                if latex_code:
                    formula = Formula(
                        latex=latex_code,
                        confidence=0.9,
                        source="text",
                        formula_type=formula_type,
                        variables=self._extract_variables(latex_code)
                    )
                    formulas.append(formula)

        # Extract inline formulas
        inline_matches = re.finditer(inline_pattern, text)
        for match in inline_matches:
            latex_code = match.group(1).strip()
            if self._is_valid_latex(latex_code):
                formula = Formula(
                    latex=latex_code,
                    confidence=0.85,
                    source="text",
                    formula_type="inline",
                    variables=self._extract_variables(latex_code)
                )
                formulas.append(formula)

        return formulas

    def _extract_ascii_formulas(self, text: str) -> List[Formula]:
        """Extract ASCII math notation formulas"""
        formulas = []

        # Common ASCII math patterns
        patterns = [
            # Equations with =
            r'([a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[^,;]+)',
            # Inequalities
            r'([a-zA-Z_][a-zA-Z0-9_]*\s*[<>≤≥]\s*[^,;]+)',
            # Function definitions
            r'(f\([^)]+\)\s*=\s*[^,;]+)',
            # Summations (ASCII style)
            r'(sum_[^_]+_[^_]+\s*[^,;]+)',
            # Integrals (ASCII style)
            r'(int_[^_]+\^[^_]+\s*[^,;]+)',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                ascii_formula = match.group(1).strip()

                # Convert to LaTeX if possible
                latex = self._ascii_to_latex(ascii_formula)

                formula = Formula(
                    latex=latex,
                    ascii_math=ascii_formula,
                    confidence=0.7,
                    source="text",
                    formula_type="equation",
                    variables=self._extract_variables(latex)
                )
                formulas.append(formula)

        return formulas

    def _extract_unicode_formulas(self, text: str) -> List[Formula]:
        """Extract Unicode mathematical notation"""
        formulas = []

        # Unicode math symbols
        unicode_math_pattern = r'[∀∃∈∉⊂⊃⊆⊇∪∩∧∨¬→↔≡≈≠≤≥∞∑∏∫∂∇±√∛∜]'

        # Find regions with high density of math symbols
        lines = text.split('\n')
        for line in lines:
            if re.search(unicode_math_pattern, line):
                # Check if line has enough math content
                math_symbols = re.findall(unicode_math_pattern, line)
                if len(math_symbols) >= 2 or any(sym in line for sym in ['=', '<', '>', '≤', '≥']):
                    # Convert Unicode to LaTeX
                    latex = self._unicode_to_latex(line)

                    formula = Formula(
                        latex=latex,
                        confidence=0.6,
                        source="text",
                        formula_type="unicode",
                        variables=self._extract_variables(latex)
                    )
                    formulas.append(formula)

        return formulas

    def _extract_with_deepseek(self, image_path: Path) -> List[Formula]:
        """Extract formulas using DeepSeek-OCR"""
        formulas = []

        # This is a placeholder for DeepSeek-OCR integration
        # In production, you would:
        # 1. Load the DeepSeek-OCR model
        # 2. Process the image
        # 3. Extract LaTeX formulas

        self.logger.info("DeepSeek-OCR extraction (placeholder)")

        # Placeholder implementation
        # In reality, this would call DeepSeek-OCR API or local model
        """
        # Example DeepSeek integration:
        from deepseek_ocr import DeepSeekOCR

        ocr = DeepSeekOCR()
        result = ocr.process(image_path)

        for formula_data in result.formulas:
            formula = Formula(
                latex=formula_data['latex'],
                confidence=formula_data['confidence'],
                source="deepseek",
                image=self._load_image_bytes(image_path)
            )
            formulas.append(formula)
        """

        return formulas

    def _extract_with_basic_ocr(self, image_path: Path) -> List[Formula]:
        """Extract formulas using basic OCR with preprocessing"""
        formulas = []

        try:
            # Load and preprocess image
            image = Image.open(image_path)

            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')

            # Enhance contrast for better OCR
            image_array = np.array(image)
            enhanced = self._enhance_math_image(image_array)

            # Save enhanced image temporarily
            temp_path = Path("/tmp/enhanced_formula.png")
            Image.fromarray(enhanced).save(temp_path)

            # Here you would use Tesseract or another OCR engine
            # This is a simplified placeholder
            self.logger.info("Basic OCR extraction (placeholder)")

            """
            # Example with pytesseract:
            import pytesseract

            text = pytesseract.image_to_string(
                temp_path,
                config='--psm 6'  # Uniform block of text
            )

            # Extract formulas from OCR text
            extracted = self.extract_formulas_from_text(text)
            for formula in extracted:
                formula.source = "ocr"
                formula.confidence *= 0.7  # Lower confidence for OCR
                formulas.append(formula)
            """

        except Exception as e:
            self.logger.error(f"Basic OCR error: {e}")

        return formulas

    def _enhance_math_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image for better math OCR"""
        # Apply adaptive thresholding
        from scipy import ndimage

        # Denoise
        denoised = ndimage.median_filter(image, size=3)

        # Increase contrast
        min_val = np.min(denoised)
        max_val = np.max(denoised)
        if max_val > min_val:
            enhanced = ((denoised - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            enhanced = denoised

        # Binarize for math symbols
        threshold = np.mean(enhanced)
        binary = np.where(enhanced < threshold, 0, 255).astype(np.uint8)

        return binary

    def _is_valid_latex(self, latex: str) -> bool:
        """Check if string is valid LaTeX formula"""
        # Check for LaTeX commands
        if re.search(r'\\[a-zA-Z]+', latex):
            return True

        # Check for math symbols
        if re.search(r'[_^{}]', latex):
            return True

        # Check for variables and operators
        if re.search(r'[a-zA-Z]\s*[=<>+\-*/]', latex):
            return True

        # Too short to be meaningful
        if len(latex) < 3:
            return False

        return True

    def _extract_variables(self, latex: str) -> List[str]:
        """Extract variable names from LaTeX formula"""
        variables = set()

        # Remove LaTeX commands
        cleaned = re.sub(r'\\[a-zA-Z]+', ' ', latex)

        # Extract single letter variables
        single_vars = re.findall(r'\b[a-zA-Z]\b', cleaned)
        variables.update(single_vars)

        # Extract subscripted variables (e.g., x_i, a_{ij})
        subscripted = re.findall(r'([a-zA-Z])_(?:\{([^}]+)\}|([a-zA-Z0-9]))', latex)
        for var, sub1, sub2 in subscripted:
            if sub1:
                variables.add(f"{var}_{sub1}")
            elif sub2:
                variables.add(f"{var}_{sub2}")

        # Extract Greek letters
        greek = re.findall(r'\\(alpha|beta|gamma|delta|epsilon|theta|lambda|mu|nu|pi|sigma|phi|psi|omega)', latex)
        variables.update(greek)

        return sorted(list(variables))

    def _ascii_to_latex(self, ascii_formula: str) -> str:
        """Convert ASCII math to LaTeX"""
        latex = ascii_formula

        # Replace common ASCII patterns
        replacements = [
            (r'\^', '^'),
            (r'_', '_'),
            (r'sqrt\(([^)]+)\)', r'\\sqrt{\1}'),
            (r'sum_([^_]+)_([^\s]+)', r'\\sum_{\1}^{\2}'),
            (r'int_([^_]+)\^([^\s]+)', r'\\int_{\1}^{\2}'),
            (r'<=', r'\\leq'),
            (r'>=', r'\\geq'),
            (r'!=', r'\\neq'),
            (r'->', r'\\to'),
            (r'<->', r'\\leftrightarrow'),
            (r'infinity', r'\\infty'),
        ]

        for pattern, replacement in replacements:
            latex = re.sub(pattern, replacement, latex)

        return latex

    def _unicode_to_latex(self, text: str) -> str:
        """Convert Unicode math symbols to LaTeX"""
        latex = text

        # Unicode to LaTeX mappings
        unicode_map = {
            '∀': r'\forall',
            '∃': r'\exists',
            '∈': r'\in',
            '∉': r'\notin',
            '⊂': r'\subset',
            '⊃': r'\supset',
            '⊆': r'\subseteq',
            '⊇': r'\supseteq',
            '∪': r'\cup',
            '∩': r'\cap',
            '∧': r'\land',
            '∨': r'\lor',
            '¬': r'\neg',
            '→': r'\to',
            '↔': r'\leftrightarrow',
            '≡': r'\equiv',
            '≈': r'\approx',
            '≠': r'\neq',
            '≤': r'\leq',
            '≥': r'\geq',
            '∞': r'\infty',
            '∑': r'\sum',
            '∏': r'\prod',
            '∫': r'\int',
            '∂': r'\partial',
            '∇': r'\nabla',
            '±': r'\pm',
            '√': r'\sqrt',
        }

        for unicode_char, latex_cmd in unicode_map.items():
            latex = latex.replace(unicode_char, latex_cmd)

        return latex

    def _deduplicate_formulas(self, formulas: List[Formula]) -> List[Formula]:
        """Remove duplicate formulas"""
        unique_formulas = {}

        for formula in formulas:
            # Use normalized LaTeX as key
            key = re.sub(r'\s+', '', formula.latex.lower())

            if key not in unique_formulas:
                unique_formulas[key] = formula
            else:
                # Keep the one with higher confidence
                if formula.confidence > unique_formulas[key].confidence:
                    unique_formulas[key] = formula

        return list(unique_formulas.values())

    def _check_deepseek_availability(self) -> bool:
        """Check if DeepSeek-OCR is available"""
        if not settings.documents.deepseek_ocr_endpoint:
            return False

        # Try to import DeepSeek-OCR module
        try:
            # This is where you would check for DeepSeek-OCR
            # import deepseek_ocr
            return False  # Placeholder - set to False for now
        except ImportError:
            return False

    def _compile_latex_patterns(self) -> Dict[str, re.Pattern]:
        """Compile common LaTeX command patterns"""
        patterns = {
            'fraction': re.compile(r'\\frac\{([^}]+)\}\{([^}]+)\}'),
            'sqrt': re.compile(r'\\sqrt(?:\[([^\]]+)\])?\{([^}]+)\}'),
            'matrix': re.compile(r'\\begin\{[pvb]?matrix\}(.*?)\\end\{[pvb]?matrix\}', re.DOTALL),
            'cases': re.compile(r'\\begin\{cases\}(.*?)\\end\{cases\}', re.DOTALL),
            'subscript': re.compile(r'_\{([^}]+)\}|_([a-zA-Z0-9])'),
            'superscript': re.compile(r'\^\{([^}]+)\}|\^([a-zA-Z0-9])'),
        }
        return patterns

    def _create_symbol_map(self) -> Dict[str, str]:
        """Create mathematical symbol mapping"""
        return {
            'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ',
            'epsilon': 'ε', 'theta': 'θ', 'lambda': 'λ', 'mu': 'μ',
            'pi': 'π', 'sigma': 'σ', 'phi': 'φ', 'psi': 'ψ', 'omega': 'ω',
            'sum': '∑', 'product': '∏', 'integral': '∫',
            'partial': '∂', 'nabla': '∇', 'infinity': '∞',
        }

    def _load_image_bytes(self, image_path: Path) -> bytes:
        """Load image as bytes"""
        with open(image_path, 'rb') as f:
            return f.read()

    def classify_formula(self, formula: Formula) -> str:
        """Classify the type of mathematical formula"""
        latex = formula.latex.lower()

        # Check for specific patterns
        if '=' in latex and not any(op in latex for op in ['<', '>', '≤', '≥']):
            if re.search(r'f\([^)]+\)\s*=', latex):
                return "function_definition"
            elif re.search(r'\\frac|/|\\div', latex):
                return "equation_rational"
            elif re.search(r'\^2|\^3|\\sqrt', latex):
                return "equation_algebraic"
            elif re.search(r'\\int|\\sum|\\prod', latex):
                return "equation_integral"
            else:
                return "equation"

        elif any(op in latex for op in ['<', '>', '≤', '≥', '\\leq', '\\geq']):
            return "inequality"

        elif re.search(r'\\lim', latex):
            return "limit"

        elif re.search(r'\\frac\{d\}|\\partial|\'', latex):
            return "derivative"

        elif re.search(r'\\begin\{.*matrix\}', latex):
            return "matrix"

        else:
            return "expression"

    def extract_formula_components(self, formula: Formula) -> Dict[str, Any]:
        """Extract components from a formula"""
        components = {
            'variables': formula.variables or [],
            'operators': [],
            'functions': [],
            'constants': [],
            'type': self.classify_formula(formula)
        }

        latex = formula.latex

        # Extract operators
        operators = re.findall(r'[+\-*/=<>]|\\(times|div|pm|mp|cdot)', latex)
        components['operators'] = list(set(operators))

        # Extract functions
        functions = re.findall(r'\\(sin|cos|tan|log|ln|exp|sqrt|lim|sum|prod|int)', latex)
        components['functions'] = list(set(functions))

        # Extract numeric constants
        constants = re.findall(r'\b\d+\.?\d*\b', latex)
        components['constants'] = constants

        return components


# Example usage
if __name__ == "__main__":
    extractor = FormulaExtractor()

    # Test text extraction
    test_text = """
    The quadratic formula is $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$.

    For any $n ∈ ℕ$, we have:
    $$\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}$$

    The Riemann Hypothesis states that all non-trivial zeros of ζ(s) have real part 1/2.
    """

    formulas = extractor.extract_formulas_from_text(test_text)

    for i, formula in enumerate(formulas):
        print(f"\nFormula {i+1}:")
        print(f"  LaTeX: {formula.latex}")
        print(f"  Type: {formula.formula_type}")
        print(f"  Variables: {formula.variables}")
        print(f"  Confidence: {formula.confidence}")

        # Classify and extract components
        classification = extractor.classify_formula(formula)
        components = extractor.extract_formula_components(formula)
        print(f"  Classification: {classification}")
        print(f"  Components: {components}")