import openai
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from decouple import config

# Системный промт для задания контекста модели
# TODO: Настроить SYSTEM PROMPT
SYSTEM_PROMPT = """
"""

import re
import pandas as pd


class LLMOutputParser:
    def __init__(self, llm_output):
        self.llm_output = llm_output

    def parse_output(self):
        """
        Разбираем вывод LLM и возвращаем таблицу с симптомами и диагнозом
        """
        symptoms, diagnosis = self.extract_symptoms_and_diagnosis()
        return self.create_table(symptoms, diagnosis)

    def extract_symptoms_and_diagnosis(self):
        """
        Извлекаем симптомы и диагноз с использованием регулярных выражений.
        Предполагаем, что данные имеют формат:
        Симптомы: <симптомы>
        Диагноз: <диагноз>
        """
        symptoms = re.search(r"Симптомы: (.*)", self.llm_output)
        diagnosis = re.search(r"Диагноз: (.*)", self.llm_output)

        symptoms = symptoms.group(1).split(", ") if symptoms else []
        diagnosis = diagnosis.group(1) if diagnosis else "Неизвестно"

        return symptoms, diagnosis

    def create_table(self, symptoms, diagnosis):
        """
        Создаем таблицу с двумя колонками: симптомы и диагноз
        """
        data = []
        for symptom in symptoms:
            data.append([symptom, diagnosis])

        # Создаем DataFrame с использованием pandas
        df = pd.DataFrame(data, columns=["Симптомы", "Диагноз"])
        return df


# Класс для работы с индексом FAISS
class FAISSIndexer:
    def __init__(self, vector_dimension):
        self.vector_dimension = vector_dimension
        self.index = faiss.IndexFlatL2(
            vector_dimension
        )  # Индекс для поиска по L2 расстоянию

    def add_vectors(self, vectors):
        """Добавление векторов в индекс FAISS"""
        self.index.add(np.array(vectors).astype("float32"))

    def search(self, query_vector, top_k=5):
        """Поиск ближайших векторов по запросу"""
        distances, indices = self.index.search(
            np.array(query_vector).astype("float32"), top_k
        )
        return distances, indices


# Класс для работы с OpenAI API
class OpenAIResponder:
    def __init__(self, api_key):
        openai.api_key = api_key

    def generate_response(self, query, context):
        """Генерация ответа с использованием модели GPT-3 с системным промтом"""
        prompt = f"{SYSTEM_PROMPT}\nВопрос: {query}\nКонтекст: {context}\nОтвет:"
        response = openai.Completion.create(
            engine="gpt-3.5-turbo",  # Использование GPT-3.5
            prompt=prompt,
            max_tokens=150,
        )
        return response.choices[0].text.strip()


# Класс для обработки текстов и преобразования их в векторы
class TextProcessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")

    def texts_to_vectors(self, texts):
        """Преобразование текстов в векторы с использованием TF-IDF"""
        return self.vectorizer.fit_transform(texts).toarray()

    def text_to_vector(self, text):
        """Преобразование одного текста в вектор"""
        return self.vectorizer.transform([text]).toarray()[0]


# Главный класс для интеграции всех компонентов
class RAGSystem:
    def __init__(self, faiss_indexer, responder, text_processor):
        self.indexer = faiss_indexer
        self.responder = responder
        self.text_processor = text_processor

    def index_documents(self, documents):
        """Индексация документов"""
        vectors = self.text_processor.texts_to_vectors(documents)
        self.indexer.add_vectors(vectors)

    def get_relevant_documents(self, query, top_k=5):
        """Поиск релевантных документов"""
        query_vector = self.text_processor.text_to_vector(query)
        _, indices = self.indexer.search(query_vector, top_k)
        return indices

    def generate_answer(self, query, documents, top_k=5):
        """Генерация ответа на основе запроса и документов"""
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
        answer = rag_system.generate_answer(query, documents, top_k=5)
    elif text_input and documents:
        # Если есть текст и документы, работаем без системного промта или с обобщённым
        query = text_input
        answer = rag_system.generate_answer(query, documents, top_k=5)
    elif text_input and not documents:
        # Если есть только текст, используем RAG по базе данных и системный промт
        query = text_input
        context = "Пожалуйста, предоставьте ответ на основании встроенной базы данных."
        answer = responder.generate_response(query, context)
    else:
        answer = "Нет данных для анализа."

    # Вывод ответа
    print("Ответ:", answer)
