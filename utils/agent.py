import openai
from openai import OpenAI
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from decouple import config
from io import StringIO, BytesIO
import pandas as pd
import json
from utils.pdf_reader import extract_text_from_pdf
from asyncify import asyncify
from create_bot import logging

# Системный промт для задания контекста модели
SYSTEM_PROMPT = """I want you to respond to my query in JSON format with two keys: `text` and `csv`.  

1. Under the `text` key, provide a detailed textual response to the query. Write naturally and concisely.  
2. Under the `csv` key, provide tabular data formatted as a CSV with the following structure:  
   Symptom,Diagnosis
   Fever,Influenza
   Headache,Migraine
   - The first row must always contain the column headers: `Symptom` and `Diagnosis`.
   - Each subsequent row should map a symptom to a diagnosis relevant to my query.  
3. Ensure the `csv` data is returned as a string with each line separated by a newline character (`\n`).  

For example, if the query is about common symptoms and their potential diagnoses, your response should follow this format:  


{
  "text": "Common symptoms like fever and headache can have various potential diagnoses. Fever is frequently associated with infections such as influenza, while headaches are often linked to conditions like migraines or tension-type headaches. It's important to consider additional symptoms and consult a medical professional for an accurate diagnosis.",
  "csv": "Symptom,Diagnosis\nFever,Influenza\nHeadache,Migraine\nCough,Common Cold\nFatigue,Anemia\nChest Pain,Heart Attack"
}

Always adhere to this structure for your response.
"""


class LLMOutputParser:
    def __init__(self, llm_output: str, system_prompt: str) -> None:
        """
        Initialize the LLMOutputParser with the output from a language model.

        Parameters
        ----------
        llm_output : str
            The output from the language model, expected to be in JSON format.
        """
        self.llm_output = llm_output

    def parse_output(self) -> dict:
        """
        Parse the LLM output JSON string and return a dictionary.

        Returns
        -------
        dict
            A dictionary with keys 'text' and 'csv', or a fallback response in case of errors.
        """
        try:
            parsed_output = json.loads(self.llm_output)
            # Проверка обязательных ключей
            if "text" not in parsed_output or "csv" not in parsed_output:
                return {
                    "error": "Missing required keys 'text' or 'csv' in the JSON output.",
                    "text": "Ошибка: отсутствуют обязательные ключи 'text' или 'csv' в ответе модели.",
                    "csv": None,
                }
            return parsed_output
        except json.JSONDecodeError as e:
            return {
                "error": f"Invalid JSON format in LLM output: {str(e)}",
                "text": "Ошибка: не удалось обработать JSON от модели.",
                "csv": None,
            }
        except Exception as e:
            return {
                "error": f"Unexpected error while parsing LLM output: {str(e)}",
                "text": "Ошибка: произошла непредвиденная ошибка обработки JSON.",
                "csv": None,
            }

    def parse_csv_to_df(self, csv_content: str) -> tuple[pd.DataFrame, str]:
        """
        Converts a CSV string into a pandas DataFrame.

        Parameters
        ----------
        csv_content : str
            CSV string content.

        Returns
        -------
        tuple
            (DataFrame, error_message). If parsing is successful, error_message is None.
        """
        try:
            data = StringIO(csv_content.strip())
            df = pd.read_csv(data)
            # Save the DataFrame to a CSV file
            df.to_csv(data, index=False, encoding="utf-8")
            return data, None
        except Exception as e:
            return None, f"Error parsing or saving CSV content: {str(e)}"

    def save_csv_to_buffer(self, csv_content: str) -> BytesIO:
        """
        Save the CSV content as a file-like object in memory.

        Parameters
        ----------
        csv_content : str
            CSV string content.

        Returns
        -------
        BytesIO
            A file-like object containing the CSV content, or None in case of error.
        """
        buffer, error = self.parse_csv_to_df(csv_content)
        if error:
            return None  # Возвращаем None, если CSV невалиден
        return buffer


# Класс для работы с индексом FAISS
class FAISSIndexer:
    def __init__(self, vector_dimension: int) -> None:
        """
        Initialize the FAISSIndexer with the given vector dimension.

        Parameters
        ----------
        vector_dimension : int
            The dimensionality of the vectors to be indexed

        Returns
        -------
        None
        """
        self.vector_dimension = vector_dimension
        self.index = faiss.IndexFlatL2(
            vector_dimension
        )  # Индекс для поиска по L2 расстоянию

    def add_vectors(self, vectors) -> None:
        """
        Add a list of vectors to the index.

        Parameters
        ----------
        vectors : List[np.ndarray]
            The list of vectors to add to the index. Each vector should be a
            1D numpy array of the same length as the index's vector dimension.

        Returns
        -------
        None
        """

        self.index.add(np.array(vectors).astype("float32"))

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for the top-k most similar vectors to the given query vector.

        Parameters
        ----------
        query_vector : np.ndarray
            The vector to search for
        top_k : int, optional
            The number of most similar vectors to return. Defaults to 5.

        Returns
        -------
        distances : np.ndarray
            The L2 distances between the query vector and the top-k most
            similar vectors
        indices : np.ndarray
            The indices of the top-k most similar vectors in the index
        """
        distances, indices = self.index.search(
            np.array(query_vector).astype("float32"), top_k
        )
        return distances, indices


# Класс для работы с OpenAI API
class OpenAIResponder:
    def __init__(self, api_key: str) -> None:
        """
        Initialize the OpenAIResponder with the given API key.

        Parameters
        ----------
        api_key : str
            The OpenAI API key to use for generating responses
        """
        openai.api_key = api_key
        self.client = OpenAI()

    def generate_response(
        self, query: str, context: str, model: str = "o1-preview", mode: str = "default"
    ) -> str:
        """
        Generates a response using the OpenAI API based on the provided query and context.

        Parameters
        ----------
        query : str
            The user's question or query to be answered by the model.
        context : str
            Additional information or context to be provided to the model for generating a response.
        model : str, optional
            The model identifier to use for generating responses. Defaults to "o1-preview".
        mode : str, optional
            The mode of operation. If set to "system", the system prompt is included in the messages.
            Defaults to "default".

        Returns
        -------
        str
            The generated response from the OpenAI model, stripped of any leading or trailing spaces.
        """
        messages = []

        if mode == "system":
            prompt_message = f"Промпт: {SYSTEM_PROMPT}\n"

        if context:
            messages.append(
                {
                    "role": "user",
                    "content": f"{prompt_message}Вопрос: {query}\nКонтекст: {context}",
                }
            )
        else:
            messages.append({"role": "user", "content": f"{query}"})

        try:
            logging.debug(f"Sending ChatGPT request: {messages}")

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
            )

            result = response.choices[0].message.content.strip()

            logging.debug(f"Got ChatGPT response: {result}")

            return result
        except openai.PermissionDeniedError as error:
            if (
                error.code == 403
                and error.error.get("code") == "unsupported_country_region_territory"
            ):
                logging.error(
                    "Your country is not supported by the OpenAI API. Please switch to a different API provider or use a VPN."
                )


# Класс для обработки текстов и преобразования их в векторы
class TextProcessor:
    def __init__(self) -> None:
        """
        Initialize the TextProcessor.

        The TextProcessor is initialized with a TfidfVectorizer that is
        configured to ignore English stop words.
        """
        self.vectorizer = TfidfVectorizer(stop_words="english")

    def texts_to_vectors(self, texts: list[str]) -> list[np.ndarray]:
        """
        Convert a list of texts into a list of vectors.

        Parameters
        ----------
        texts : list of str
            The list of texts to convert into vectors

        Returns
        -------
        vectors : list of np.ndarray
            The list of vectors, where each vector is a 2D numpy array
            representing the corresponding text.
        """
        logging.debug(f"Got texts type: {type(texts)}")
        logging.debug(f"Got texts: {texts}")
        chunk_size = 512  # Размер чанка в символах
        self.chunks = []
        for text in texts:
            for i in range(0, len(text), chunk_size):
                chunk = text[i : i + chunk_size]
                self.chunks.append(chunk)

        matrix = self.vectorizer.fit_transform(self.chunks)

        logging.debug(f"Got matrix type: {type(matrix)}")
        logging.debug(f"Got matrix: {matrix}")
        vectors = matrix.toarray()
        logging.debug(f"Got vectors type: {type(vectors)}")
        logging.debug(f"Got vectors: {vectors}")
        return vectors

    def text_to_vector(self, text: str) -> np.ndarray:
        """
        Convert a single text into a vector.

        Parameters
        ----------
        text : str
            The text to convert into a vector

        Returns
        -------
        vector : np.ndarray
            The vector representation of the text, as a 1D numpy array.
        """
        return self.vectorizer.transform([text])


# Главный класс для интеграции всех компонентов
class RAGSystem:
    def __init__(
        self,
        faiss_indexer: FAISSIndexer,
        responder: OpenAIResponder,
        text_processor: TextProcessor,
    ) -> None:
        """
        Initialize the RAGSystem with the given components.

        Parameters
        ----------
        faiss_indexer : FAISSIndexer
            The FAISSIndexer object to use for indexing and searching vectors.
        responder : OpenAIResponder
            The OpenAIResponder object to use for generating responses.
        text_processor : TextProcessor
            The TextProcessor object to use for converting texts into vectors.

        Returns
        -------
        None
        """
        self.responder = responder
        self.text_processor = text_processor

    def index_documents(self, documents: list[str]) -> None:
        """
        Index a list of documents by converting them into vectors and adding them to the FAISS indexer.

        Parameters
        ----------
        documents : list of str
            The list of documents to be indexed. Each document is expected to be a string.

        Returns
        -------
        None
        """
        vectors = self.text_processor.texts_to_vectors(documents)
        vector_dimension = vectors[0].shape[0]
        logging.debug(f"Got vector dimension: {vector_dimension}")
        self.indexer = FAISSIndexer(vector_dimension=vector_dimension)
        self.indexer.add_vectors(vectors)

    def get_relevant_documents(self, query: str, top_k: int = 5) -> np.ndarray:
        """
        Retrieve the indices of the top-k most relevant documents for the given query.

        Parameters
        ----------
        query : str
            The query string for which to find the most relevant documents.
        top_k : int, optional
            The number of top relevant documents to retrieve. Defaults to 5.

        Returns
        -------
        indices : np.ndarray
            The indices of the top-k most relevant documents in the indexed collection.
        """
        query_vector = self.text_processor.text_to_vector(query)
        _, indices = self.indexer.search(query_vector.toarray(), top_k)
        return indices

    def generate_answer(
        self, query: str, documents: list[str], mode: str = "default", top_k: int = 5
    ) -> str:
        """
        Generate an answer to the given query by finding the most relevant documents in the given collection,
        creating a context from the top-k most relevant documents, and then using the OpenAI API to generate a response.

        Parameters
        ----------
        query : str
            The query string for which to find the most relevant documents and generate a response.
        documents : list of str
            The list of documents to search for the most relevant ones.
        mode : str, optional
            The mode of operation. If set to "system", the system prompt is included in the messages.
            Defaults to "default".
        top_k : int, optional
            The number of top relevant documents to retrieve. Defaults to 5.

        Returns
        -------
        str
            The generated response from the OpenAI model, stripped of any leading or trailing spaces.
        """
        relevant_indices = self.get_relevant_documents(query, top_k)
        logging.debug(f"Got relevant indices: {relevant_indices}")

        context = "\n".join(
            [self.text_processor.chunks[i] for i in relevant_indices[0]]
        )

        logging.debug(f"Got context: {context}")

        return self.responder.generate_response(query=query, context=context, mode=mode)


class LLMAgent:
    def __init__(self, api_key: str, vector_dimension: int = 1):
        self.previous_documents = None

        # Инициализация компонентов
        self.faiss_indexer = FAISSIndexer(vector_dimension=vector_dimension)
        self.responder = OpenAIResponder(api_key=api_key)
        self.text_processor = TextProcessor()

        # Инициализация RAG-системы
        self.rag_system = RAGSystem(
            self.faiss_indexer, self.responder, self.text_processor
        )

    @asyncify
    def process(self, text_input: str = None, documents: BytesIO = None):
        documents = extract_text_from_pdf(documents)
        if documents:
            # Если есть документы, индексируем их и сохраняем в переменную
            self.rag_system.index_documents(documents)
            self.previous_documents = documents
        else:
            # Если нет документов, используем ранее загруженные документы
            documents = self.previous_documents

        # Основной процесс с проверками
        if documents:
            # Если текста нет, обобщаем документы
            logging.debug("Обнаружен документ, входного текста не найдено")
            query = "Обобщите основные заболевания из документов, ответ предоставьте в формате CSV с двумя колонками: симптомы и диагноз."
            answer = self.rag_system.generate_answer(query, documents, mode="system")
        elif text_input:
            logging.debug("Обнаружен входной текст")
            query = text_input
            answer = self.rag_system.generate_answer(query, documents, mode="system")
        else:
            logging.debug(
                "Текст и документы отсутствуют, возвращаем сообщение об ошибке"
            )
            # Если нет текста и документов, возвращаем сообщение об ошибке
            return {
                "text": "",
                "csv": "",
                "error": "Ошибка: текст и документы отсутствуют.",
            }

        # Парсим и обрабатываем результат
        output_parser = LLMOutputParser(answer, SYSTEM_PROMPT)
        parsed_output = output_parser.parse_output()

        # Обрабатываем результат с учетом ошибок
        response = {
            "text": parsed_output.get("text", ""),
            "csv_buffer": None,
            "error": parsed_output.get("error", None),
        }

        if parsed_output.get("csv", None) != None:
            # Если CSV присутствует, сохраняем в буфер
            csv_buffer = parsed_output["csv"]
            if csv_buffer:
                response["csv_buffer"] = csv_buffer

        logging.debug(f"Got model response: {response}")

        return response


if __name__ == "__main__":
    api_key = config("OPENAI_API_KEY")
    agent = LLMAgent(api_key=api_key)

    response = agent.process(
        text_input="Какие симптомы характерны для вирусных заболеваний?",
        documents=[
            "Симптомы гриппа включают высокую температуру.",
            "Мигрень вызывает сильную головную боль.",
        ],
    )

    if response.get("error"):
        print("Errors occurred during processing:")
        print(response["error"])
    else:
        print("Text Response:")
        print(response["text"])

        if response["csv_buffer"]:
            print("\nCSV Content:")
            csv_df = pd.read_csv(response["csv_buffer"])
            print(csv_df)
