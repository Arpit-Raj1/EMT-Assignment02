"""
utils_io.py
Utilities for figure/model saving and PDF report generation.
"""

import os
import matplotlib.pyplot as plt
from typing import Any

__all__ = ["save_figure", "save_model", "build_pdf_report"]


def save_figure(fig: plt.Figure, path: str, dpi: int = 200) -> None:
    """
    Save a matplotlib figure to file (PDF/PNG).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def save_model(model: Any, path: str) -> None:
    """
    Save a scikit-learn model to file.
    """
    import joblib

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def build_pdf_report(notebook_path: str, pdf_path: str) -> None:
    """
    Convert a notebook to PDF using nbconvert and WeasyPrint.
    """
    import subprocess
    import shutil

    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    html_path = pdf_path.replace(".pdf", ".html")
    try:
        subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "html",
                "--output",
                html_path,
                notebook_path,
            ],
            check=True,
        )
        subprocess.run(["weasyprint", html_path, pdf_path], check=True)
        print(f"PDF generated with WeasyPrint at {pdf_path}")
    except Exception as weasy_err:
        print(
            "WeasyPrint PDF export failed, trying nbconvert PDF export (requires TeX/LaTeX)..."
        )
        try:
            subprocess.run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "pdf",
                    "--output",
                    os.path.basename(pdf_path),
                    notebook_path,
                ],
                check=True,
            )
            # Move the PDF to the desired location if needed
            generated_pdf = os.path.splitext(notebook_path)[0] + ".pdf"
            if os.path.exists(generated_pdf):
                shutil.move(generated_pdf, pdf_path)
            print(f"PDF generated with nbconvert at {pdf_path}")
        except Exception as nbconvert_err:
            print("Both WeasyPrint and nbconvert PDF export failed.")
            print("WeasyPrint error:", weasy_err)
            print("nbconvert error:", nbconvert_err)
            raise RuntimeError(
                "PDF export failed. Please ensure WeasyPrint or TeX/LaTeX is installed."
            )
