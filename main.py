import argparse
import os
import glob

from preprocess.preprocess import PDFPreprocessor
from retrieval.retrieval import Retriever
from llm.llm import GeminiLLM
from agents.retriever_agent import RetrieverAgent
from agents.image_analyst_agent import ImageAnalystAgent
from agents.validator_agent import ValidatorAgent
from agents.orchestrator_agent import OrchestratorAgent


def main():
    parser = argparse.ArgumentParser(description="Agentic Document QA System")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--question", required=True, help="Question to ask")

    args = parser.parse_args()

    # Step 0: resolve pdf path(s) - single file or all pdfs in a directory
    if os.path.isdir(args.pdf):
        pdf_paths = sorted(glob.glob(os.path.join(args.pdf, "*.pdf")))
        if not pdf_paths:
            print(f"No PDF files found in directory: {args.pdf}")
            return
    else:
        pdf_paths = [args.pdf]

    print(f"  -> Found {len(pdf_paths)} PDF(s): {[os.path.basename(p) for p in pdf_paths]}")

    # Init shared LLM
    llm = GeminiLLM()

    # Step 1: check if merged index is still valid for all pdfs
    retriever = Retriever()
    if retriever.load_merged_if_valid(pdf_paths):
        print("[1/5] Merged index valid, skipping preprocessing...")
        print("[2/5] Skipped (cached)")
        print("[3/5] Skipped (cached)")
    else:
        all_text_chunks = []
        all_image_chunks = []
        preprocessor = PDFPreprocessor()

        for pdf_path in pdf_paths:
            pdf_name = os.path.basename(pdf_path)

            # check per-pdf cache first
            cached = retriever.load_pdf_chunks(pdf_path)
            if cached:
                text_chunks, image_chunks = cached
                print(f"[1/5] {pdf_name} loaded from cache ({len(text_chunks)} text, {len(image_chunks)} images)")
            else:
                print(f"[1/5] Preprocessing {pdf_name}...")
                text_chunks, image_chunks = preprocessor(pdf_path)
                print(f"  -> {len(text_chunks)} text chunks, {len(image_chunks)} images found")

                # Step 2: describe images with Gemini
                if image_chunks:
                    print(f"[2/5] Describing images for {pdf_name}...")
                    for chunk in image_chunks:
                        print(f"  -> Describing {chunk['image_path']}...")
                        try:
                            chunk["description"] = llm.describe_image(chunk["image_path"])
                        except Exception as e:
                            print(f"  -> Failed: {e}")
                            chunk["description"] = None

                # save per-pdf cache
                retriever.save_pdf_chunks(pdf_path, text_chunks, image_chunks)
                print(f"  -> Cached {pdf_name}")

            all_text_chunks.extend(text_chunks)
            all_image_chunks.extend(image_chunks)

        # Step 3: build merged FAISS indexes from all pdfs
        print(f"[3/5] Building merged FAISS indexes ({len(all_text_chunks)} text, {len(all_image_chunks)} images)...")
        retriever.index_text(all_text_chunks)
        retriever.index_images(all_image_chunks)
        retriever.save_merged_stamp(pdf_paths)

        text_count = retriever.text_index.ntotal if retriever.text_index else 0
        image_count = retriever.image_index.ntotal if retriever.image_index else 0
        print(f"  -> Text index: {text_count} vectors")
        print(f"  -> Image index: {image_count} vectors")

    # Step 4: setup agents (all share the same llm instance)
    print("[4/5] Setting up agents...")
    retriever_agent = RetrieverAgent(retriever)
    analyst_agent = ImageAnalystAgent(llm)
    validator_agent = ValidatorAgent(llm)
    orchestrator = OrchestratorAgent(retriever_agent, analyst_agent, validator_agent)

    # Step 5: run question through orchestrator
    print(f"[5/5] Answering: '{args.question}'")
    print(f"\n{'=' * 60}")

    try:
        answer = orchestrator(args.question)
        print("ANSWER")
        print(f"{'=' * 60}")
        print(answer)
    except Exception as e:
        print(f"Error: {e}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()