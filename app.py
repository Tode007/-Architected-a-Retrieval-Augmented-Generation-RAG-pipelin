import os
from flask import Flask, request, jsonify
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

app = Flask(__name__)


# --- 1. ARCHITECTING THE RAG PIPELINE ---

INTERNAL_KNOWLEDGE_BASE = [
    "Data Scientist Track: Required skills include Python, SQL, Machine Learning, Scikit-learn, and RAG architectures.",
    "Data Analyst Track: Required skills include Excel, SQL, Power BI, DAX, Tableau, and KPI Dashboarding.",
    "AI Engineer Track: Required skills include PyTorch, LLMs, LangChain, Vector Databases, and Prompt Engineering."
]

# Create an in-memory vector store using FAISS and OpenAI Embeddings
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts(INTERNAL_KNOWLEDGE_BASE, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 1})


# --- 2. DEFINING STRUCTURED OUTPUT SCHEMAS ---
class CareerAnalysis(BaseModel):
    target_role: str = Field(description="The matching career track path found")
    matched_skills: List[str] = Field(description="Skills the candidate possesses")
    missing_skills: List[str] = Field(description="Skills required but missing from the resume")
    custom_interview_questions: List[str] = Field(description="3 tailored technical interview questions based strictly on their existing skills")


# --- 3. CORE PROCESSING API ENDPOINT ---
@app.route('/api/analyze-resume', methods=['POST'])
def analyze_resume():
    try:
        data = request.get_json()
        resume_context = data.get('resume_text', '')
        requested_track = data.get('target_track', 'Data Scientist')

        if not resume_context:
            return jsonify({"error": "No resume text provided"}), 400

        # Step A: Retrieve grounding facts from internal knowledge base to eliminate hallucination
        docs = retriever.invoke(requested_track)
        grounding_knowledge = docs[0].page_content

        # Step B: Construct a precise prompt template passing both contexts
        parser = JsonOutputParser(pydantic_object=CareerAnalysis)
        
        prompt = ChatPromptTemplate.from_template(
            "You are an expert AI Career Guidance System.\n"
            "Use the following verified internal knowledge base text to evaluate the candidate's roadmap. "
            "Do not hallucinate skills not present in the internal path:\n"
            "-----\n"
            "INTERNAL PATHWAY FACTS:\n{knowledge}\n"
            "-----\n"
            "CANDIDATE RESUME CONTEXT:\n{resume}\n"
            "-----\n"
            "Instructions:\n"
            "1. Evaluate the candidate against the pathway.\n"
            "2. Generate a skill-gap matrix identifying what they have and what they lack.\n"
            "3. Generate 3 challenging technical interview questions targeting the candidate's listed skills to verify depth.\n\n"
            "{format_instructions}"
        )

        # Build the chain using modern LangChain Expression Language (LCEL)
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1) # Low temperature to minimize hallucinations
        chain = prompt | model | parser

        # Execute chain
        response_payload = chain.invoke({
            "knowledge": grounding_knowledge,
            "resume": resume_context,
            "format_instructions": parser.get_format_instructions()
        })

        return jsonify(response_payload), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
