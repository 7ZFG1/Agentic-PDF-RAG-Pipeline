import os
import glob
import argparse
from typing import List, Dict, Tuple
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Importing RAG pipeline components from your project structure
from preprocess.preprocess import PDFPreprocessor
from retrieval.retrieval import Retriever
from llm.llm import GeminiLLM
from agents.retriever_agent import RetrieverAgent
from agents.image_analyst_agent import ImageAnalystAgent
from agents.validator_agent import ValidatorAgent
from agents.orchestrator_agent import OrchestratorAgent

# Importing the Judge class from the previous script
from evaluator import LLMJudge


def setup_rag_pipeline(pdf_source: str) -> Tuple[OrchestratorAgent, GeminiLLM]:
    """
    Initializes the document indexing and agent architecture.
    Extracts the setup flow from main.py so it only runs once during evaluation.
    """
    print(f"\n[SYSTEM] Initializing RAG Pipeline with data source: '{pdf_source}'")
    
    # Step 0: Resolve PDF path(s)
    if os.path.isdir(pdf_source):
        pdf_paths = sorted(glob.glob(os.path.join(pdf_source, "*.pdf")))
        if not pdf_paths:
            raise FileNotFoundError(f"No PDF files found in directory: {pdf_source}")
    else:
        pdf_paths = [pdf_source]

    # Initialize shared LLM
    llm = GeminiLLM()

    # Step 1-3: Indexing and Retrieval Setup
    retriever = Retriever()
    if retriever.load_merged_if_valid(pdf_paths):
        print("[SYSTEM] Merged index valid, skipping preprocessing...")
    else:
        all_text_chunks = []
        all_image_chunks = []
        preprocessor = PDFPreprocessor()

        for pdf_path in pdf_paths:
            pdf_name = os.path.basename(pdf_path)
            cached = retriever.load_pdf_chunks(pdf_path)
            
            if cached:
                text_chunks, image_chunks = cached
                print(f"[SYSTEM] {pdf_name} loaded from cache.")
            else:
                print(f"[SYSTEM] Preprocessing {pdf_name}...")
                text_chunks, image_chunks = preprocessor(pdf_path)

                if image_chunks:
                    print(f"[SYSTEM] Describing images for {pdf_name}...")
                    for chunk in image_chunks:
                        try:
                            chunk["description"] = llm.describe_image(chunk["image_path"])
                        except Exception as e:
                            chunk["description"] = None

                retriever.save_pdf_chunks(pdf_path, text_chunks, image_chunks)

            all_text_chunks.extend(text_chunks)
            all_image_chunks.extend(image_chunks)

        print("[SYSTEM] Building merged FAISS indexes...")
        retriever.index_text(all_text_chunks)
        retriever.index_images(all_image_chunks)
        retriever.save_merged_stamp(pdf_paths)

    # Step 4: Setup Agents
    print("[SYSTEM] Setting up agents...")
    retriever_agent = RetrieverAgent(retriever)
    analyst_agent = ImageAnalystAgent(llm)
    validator_agent = ValidatorAgent(llm)
    
    orchestrator = OrchestratorAgent(retriever_agent, analyst_agent, validator_agent)
    
    print("[SYSTEM] Pipeline is ready!\n")
    return orchestrator, llm


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG System using LLM-as-a-Judge")
    parser.add_argument("--pdf", default="data", help="Path to PDF file or directory")
    args = parser.parse_args()

    # 1. Set up the entire RAG pipeline and get the orchestrator agent and shared LLM
    orchestrator, shared_llm = setup_rag_pipeline(args.pdf)

    # 2. Initialize the Judge evaluator with the shared LLM
    judge = LLMJudge(shared_llm)

    # 3. Define the Evaluation Dataset (Question & Ground Truth Reference pairs)
    evaluation_dataset: List[Dict[str, str]] = [
        {
            "question": "What is the performance metrics for the YOLOV8 (for YOLOV8n, YOLOV8s, YOLOV8m, YOLOV8l, YOLOV8x)?",
            "reference": "The mAP@0.5 scores for the different YOLOv8 versions are: YOLOv8n: 47.2%, YOLOv8s: 58.5%, YOLOv8m: 66.3%, YOLOv8l: 69.8%, and YOLOv8x: 71.5%."
        },
        {
            "question": "Ödevin mimari tasarım dökümanı nasıl olması beklenmektedir",
            "reference": """
                    Çok-modlu, uzun bağlamlı belgeler üzerinde soru-cevap yapabilen bir agentic sistem mimarisi
                    tasarlayınız. Tasarım dokümanınız aşağıdaki soruları yanıtlamalıdır:
                    1. Belge ön işleme (Document Pre-processing): Metin ve görsel içerik nasıl ayrıştırılır?
                    Hangi araçlar/kütüphaneler kullanılır?
                    2. Yapısal navigasyon: Ajan, uzun bir belgede ilgili bölümü nasıl bulur? Belge yapısı nasıl
                    temsil edilir?
                    3. Retrieval stratejisi: Metin tabanlı ve görsel tabanlı arama nasıl entegre edilir?
                    4. Ajan mimarisi: Kaç ajan olacak, rolleri nedir, birbirleriyle nasıl iletişim kurar?
                    5. Doğrulama ve güvenilirlik: Yanlış cevapların önüne nasıl geçilir?
                    6. Bellek yönetimi: Görevler arası öğrenme mümkün mü? Nasıl?
                    Tasarım dokümanı formatı: Markdown veya PDF, en az 2 sayfa, mimari diyagram içermesi beklenir (ASCII diyagram da kabul edilir).
"""
        },
        {
            "question": "What is the k, s and p value of first conv layer in backbone?",
            "reference": "In the standard YOLO architecture, the first convolutional layer typically uses a kernel size (k) of 3, stride (s) of 2, and padding (p) of 1 to perform initial downsampling."
        }
    ]

    # 4. Evaluation Loop
    total_score = 0
    results = []
    total_questions = len(evaluation_dataset)

    print("=" * 60)
    print(f"STARTING EVALUATION OVER {total_questions} QUESTIONS")
    print("=" * 60)

    for i, data in enumerate(evaluation_dataset):
        question = data["question"]
        reference = data["reference"]
        
        print(f"\n[Question {i+1}/{total_questions}]: {question}")
        
        # -- Step A: Get prediction from your Orchestrator Agent
        try:
            prediction = orchestrator(question)
        except Exception as e:
            prediction = f"Error during generation: {e}"
            print(f"  -> Agent failed: {e}")

        # -- Step B: Trigger the Judge to evaluate the prediction against the reference
        print("  -> Evaluating prediction...")
        eval_result = judge.evaluate(
            question=question,
            reference=reference,
            prediction=prediction
        )
        
        results.append(eval_result)
        total_score += eval_result["score"]
        
        print(f"  -> Judge Score: {eval_result['score']}")

    # 5. Final Report
    accuracy = (total_score / total_questions) * 100

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"System Accuracy : {accuracy:.2f}%")
    print(f"Correct Answers : {total_score} / {total_questions}")
    print("=" * 60)

    for i, res in enumerate(results):
        print(f"\n--- [Result {i+1}] ---")
        print(f"Q: {res['question']}")
        print(f"Agent Prediction: {res['prediction'][:200]}... [truncated]") 
        print(f"Judge Score: {res['score']}")
        print(f"Judge Reason:\n{res['judge_response']}")


if __name__ == "__main__":
    main()