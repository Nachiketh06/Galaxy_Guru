import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langsmith import traceable
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(dotenv_path="./.env", verbose=True, override=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@traceable(run_type="llm", metadata={"ls_provider": "ollama", "model": "mistral"})
def create_qa_agent(pdf_path, model_name="mistral"):
    """
    Create a question-answering agent for a specific PDF document.

    Args:
        pdf_path (str): Path to the PDF file
        model_name (str): Name of the Ollama model to use

    Returns:
        qa_chain: A QA chain that can answer questions about the PDF
    """
    persist_directory = "./data/chroma_db"

    # Check if the Chroma store already exists
    if os.path.exists(persist_directory):
        logging.info("Loading existing Chroma store...")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=OllamaEmbeddings(model=model_name))
    else:
        logging.info("Creating new Chroma store...")
        # 1. Load the PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        logging.info(f"Loaded {len(pages)} pages from the PDF.")

        # 2. Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        splits = text_splitter.split_documents(pages)
        logging.info(f"Split the document into {len(splits)} chunks.")

        # 3. Create embeddings and store them
        embeddings = OllamaEmbeddings(model=model_name)
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

        for i, chunk in enumerate(tqdm(splits, desc="Processing chunks"), 1):
            vectorstore.add_documents([chunk], embedding=embeddings)
        logging.info(f"Stored {len(splits)} chunks in the vectorstore.")

    # 4. Create the LLM
    llm = Ollama(model="mistral:latest")

       # 5. Create a custom prompt template with examples
    prompt_template = """
    You are an educational assistant trained to answer astronomy questions using only textbook content.

    - Use only the context below to answer the question.
    - If the answer is clearly stated OR can be reasonably inferred from the context, summarize it clearly.
    - Do not use outside knowledge.
    - If the answer truly cannot be found or inferred, respond: "I couldn’t find that in the provided textbook content."

    Examples:

    Q: What is a star?
    A: A star is a massive, luminous object that generates its own energy through nuclear fusion. Stars like the Sun produce light and heat by converting hydrogen into helium in their cores.

    Q: What is a planet?
    A: A planet is a body that orbits a star, does not produce its own light, and has cleared its orbit of other debris. It reflects the light of the star it orbits.

    Now answer the following based on the given context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # 6. Create and return the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 7}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain


@traceable(run_type="chain")
def ask_question(qa_chain, question, education_level):
    """
    Ask a question to the QA chain and get the response, dynamically adjusting the prompt for education level.
    Also generates 2-3 follow-up questions based on the original question and answer.
    """
    def create_prompt(education_level):
        if education_level == "High School":
            level_instruction = "Explain in simple, easy-to-understand language suitable for a high school student."
        elif education_level == "Graduate/Masters":
            level_instruction = "Provide a detailed, technical explanation appropriate for graduate-level study."
        else:
            level_instruction = "Explain clearly with moderate technical detail suitable for an undergraduate student."

        prompt_template = f"""
You are an educational assistant trained to answer astronomy questions using only textbook content.

{level_instruction}

- Use only the context below to answer the question.
- If the answer is clearly stated OR can be reasonably inferred, summarize it clearly.
- Do not use outside knowledge.
- If the answer truly cannot be found or inferred, respond: "I couldn’t find that in the provided textbook content."

Context:
{{context}}

Question:
{{question}}

Answer:
"""
        return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    try:
        prompt = create_prompt(education_level)
        qa_chain.combine_documents_chain.llm_chain.prompt = prompt

        response = qa_chain({"query": question})
        answer = response["result"]
        sources = [doc.page_content for doc in response["source_documents"]]

        # --- Generate follow-up questions using Ollama ---
        llm = Ollama(model="mistral:latest")  # or the same one you use for QA
        followup_prompt = f"""
You are an assistant helping students explore astronomy topics. Based on the original question and answer, generate 2–3 concise follow-up questions that would help a student dive deeper.

Original Question: {question}
Answer: {answer}

Follow-up Questions:
"""

        followup_questions = llm.invoke(followup_prompt).strip().split("\n")

        return {
            "answer": answer,
            "sources": sources,
            "follow_up_questions": [q.strip("- ").strip() for q in followup_questions if q.strip()]
        }

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return {
            "error": f"An error occurred: {str(e)}",
            "answer": None,
            "sources": None,
            "follow_up_questions": []
        }














    

