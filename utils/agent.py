import openai
from openai import OpenAI
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from decouple import config
from io import StringIO, BytesIO
import pandas as pd

# Системный промт для задания контекста модели
SYSTEM_PROMPT = """You’re an adept medical data analyst with extensive experience in extracting and organizing medical information from textual sources into structured formats. Your specialty lies in converting complex medical narratives into clear and concise CSV files that facilitate data analysis and interpretation.

Your task is to read the provided text and extract information to create a CSV file format with two columns: **Symptom** and **Diagnosis**. Follow these guidelines:  

1. Identify the **symptom(s)** described in the text and write them exactly as they appear or in concise form if too verbose.  
2. Identify the **diagnosis** mentioned and provide it in clear, specific terms.  
3. If the text contains multiple symptoms or diagnoses, map each symptom to its corresponding diagnosis as separate rows in the CSV format.  
4. If no diagnosis is mentioned or unclear, write "Not specified" under the **Diagnosis** column.  
5. Output the result as text in CSV format, like this example:  
   Symptom,Diagnosis
   Fever,Influenza
   Headache,Migraine
"""


class LLMOutputParser:
    def __init__(self, llm_output):
        """
        Initialize the LLMOutputParser with the output from a language model.

        Parameters
        ----------
        llm_output : str
            The output from the language model, expected to be in CSV format as a string.
        """
        self.llm_output = llm_output

    def parse_output(self):
        """
        Reads the LLM output as a CSV string and returns a pandas DataFrame.

        Parameters
        ----------
        None

        Returns
        -------
        df : pandas.DataFrame
            The LLM output parsed into a DataFrame
        """
        data = StringIO(self.llm_output.strip())
        df = pd.read_csv(data)
        return df

    def save_to_csv(self, df, file_path):
        """
        Saves the given DataFrame to a CSV file at the given file path.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to save to the CSV file
        file_path : str
            The path to the file to save the DataFrame to

        Returns
        -------
        str
            The contents of the CSV file as a string
        """
        buffer = BytesIO()
        # Сохраняем данные в формате CSV в буфер
        df.to_csv(buffer, index=False, encoding="utf-8")
        # Перемещаем курсор буфера в начало, чтобы данные можно было читать
        buffer.seek(0)
        return buffer.getvalue()


# Класс для работы с индексом FAISS
class FAISSIndexer:
    def __init__(self, vector_dimension):
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

    def add_vectors(self, vectors):
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

    def search(self, query_vector, top_k=5):
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
    def __init__(self, api_key):
        """
        Initialize the OpenAIResponder with the given API key.

        Parameters
        ----------
        api_key : str
            The OpenAI API key to use for generating responses
        """
        openai.api_key = api_key
        self.client = OpenAI()

    def generate_response(self, query, context, model="o1-preview", mode="default"):
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
            messages.append({"role": "system", "content": SYSTEM_PROMPT})

        messages.append(
            {"role": "user", "content": f"Вопрос: {query}\nКонтекст: {context}\nОтвет:"}
        )

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content.strip()


# Класс для обработки текстов и преобразования их в векторы
class TextProcessor:
    def __init__(self):
        """
        Initialize the TextProcessor.

        The TextProcessor is initialized with a TfidfVectorizer that is
        configured to ignore English stop words.
        """
        self.vectorizer = TfidfVectorizer(stop_words="english")

    def texts_to_vectors(self, texts):
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
        return self.vectorizer.fit_transform(texts).toarray()

    def text_to_vector(self, text):
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
        return self.vectorizer.transform([text]).toarray()[0]


# Главный класс для интеграции всех компонентов
class RAGSystem:
    def __init__(self, faiss_indexer, responder, text_processor):
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
        self.indexer = faiss_indexer
        self.responder = responder
        self.text_processor = text_processor

    def index_documents(self, documents):
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
        self.indexer.add_vectors(vectors)

    def get_relevant_documents(self, query, top_k=5):
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
        _, indices = self.indexer.search(query_vector, top_k)
        return indices

    def generate_answer(self, query, documents, mode="default", top_k=5):
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
        context = "\n".join([documents[i] for i in relevant_indices[0]])
        return self.responder.generate_response(query, context)


# Пример использования системы
if __name__ == "__main__":
    # Инициализация компонентов
    faiss_indexer = FAISSIndexer(vector_dimension=100)  # Размерность вектора
    responder = OpenAIResponder(api_key=config("OPENAI_API_KEY"))
    text_processor = TextProcessor()

    # Пример текста и документов
    text_input = "Какие симптомы ожирения?"  # Пример пользовательского текста
    documents = [
        "Мигрень часто сопровождается сильной головной болью, тошнотой и чувствительностью к свету.",
        "Ожирение может привести к различным заболеваниям, таким как диабет 2 типа и гипертония.",
        "Пневмония характеризуется кашлем, одышкой и болями в груди.",
    ]

    # Инициализация RAG-системы
    rag_system = RAGSystem(faiss_indexer, responder, text_processor)
    rag_system.index_documents(documents)

    # Логика выбора сценария
    if not text_input and documents:
        # Если текста нет, работаем только с документами, добавляем системный промт
        query = "Обобщите основные заболевания из документов."
        answer = rag_system.generate_answer(query, documents, mode="system")
    elif text_input and documents:
        # Если есть текст и документы, работаем без системного промта или с обобщённым
        query = text_input
        answer = rag_system.generate_answer(query, documents)
    elif text_input and not documents:
        # Если есть только текст, используем RAG по базе данных
        query = text_input
        context = "Пожалуйста, предоставьте ответ на основании встроенной базы данных."
        answer = responder.generate_response(query, context)
    else:
        answer = "Нет данных для анализа."

    # Вывод ответа
    print("Ответ:", answer)
