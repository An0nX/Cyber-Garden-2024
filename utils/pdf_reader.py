import PyPDF2
import io
from create_bot import logging


def extract_text_from_pdf(file: io.BytesIO) -> list[str]:
    """
    Extract text from a PDF file.

    Parameters
    ----------
    file : io.BytesIO
        A BytesIO object containing the PDF file.

    Returns
    -------
    str
        The text extracted from the PDF file.
    """
    logging.debug(f'Got PDF file io object: {file}')

    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += "\n" + page.extract_text()

    logging.debug(f'Extracted text from PDF: {text}')

    return [text]
