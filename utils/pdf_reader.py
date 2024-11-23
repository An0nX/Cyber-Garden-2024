import PyPDF2
import io


def extract_text_from_pdf(file: io.BytesIO) -> str:
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
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(pdf_reader.pages):
        pdf_page = pdf_reader[page_num]
        text += '\n' + pdf_page.extract_text()

    return text
