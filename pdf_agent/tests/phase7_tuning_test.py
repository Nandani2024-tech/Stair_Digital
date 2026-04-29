# tests/phase7_tuning_test.py
import sys
import os
import unittest
import re
from typing import List

# Add parent dir to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ParsedPage, Chunk
from ingestion.metadata import detect_sections
from ingestion.chunker import chunk_pages
from indexing.index_builder import build_index
from retrieval.searcher import search_document
from retrieval.hallucination_gate import evaluate_retrieval_gate

class Phase7TuningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a synthetic doc representing key parts of the RBI report
        cls.doc_id = "rbi_tune_test_fixture"
        cls.pages = [
            ParsedPage(
                page_number=1,
                raw_text="""Monetary Policy Statement: October 6, 2023. 
                In today's monetary policy meeting, the RBI kept the policy rate, the repo rate, unchanged at 6.50%. 
                Furthermore, it retained the stance of 'withdrawal of accommodation'.""",
                blocks=[],
                section_title=None
            ),
            ParsedPage(
                page_number=2,
                raw_text="""The RBI remains hawkish amid inflationary concerns. 
                As such the GDP growth projection for FY24 remained unchanged at 6.5%. 
                The RBI remains cautious about inflation and remains cognizant of the upside risks to the outlook.""",
                blocks=[
                    {"lines": [{"spans": [{"text": "Source: CMIE", "size": 12, "flags": 16}]}]}
                ],
                section_title=None
            ),
            ParsedPage(
                page_number=3,
                raw_text="""The RBI also announced various developmental and regulatory policy measures. 
                Gold Loan – Bullet Repayment Scheme – UCBs: The Urban Cooperative Banks (UCBs) have been allowed to extend the glide path 
                for the achievement of priority sector lending targets beyond March 2023. 
                The ceiling of gold loans that can be granted under the bullet repayment scheme will be increased from Rs 2 lakh to Rs 4 lakh.""",
                blocks=[],
                section_title=None
            )
        ]
        # 1. Detect sections
        cls.detected = detect_sections(cls.pages)
        
        # 2. Index the synthetic doc
        chunks = chunk_pages(cls.detected, cls.doc_id)
        build_index(chunks)
        print(f"\n[TEST] Synthetic doc {cls.doc_id} indexed.")

    def test_section_metadata_quality(self):
        """Verify that 'Source: CMIE' is NOT detected as a section title."""
        for p in self.detected:
            if p.section_title:
                self.assertNotIn("Source", p.section_title, f"Attribution '{p.section_title}' should not be a heading")
        print("[PASS] Metadata Quality: No attribution-based headings found.")

    def test_valid_queries(self):
        valid_queries = [
            "What was the repo rate decided in the October 2023 meeting?",
            "What are the GDP growth projections for FY24?",
            "Explain the new gold loan rules for Urban Cooperative Banks."
        ]
        
        print("\n--- Valid Queries ---")
        for q in valid_queries:
            with self.subTest(query=q):
                res = search_document(q, self.doc_id)
                gate = evaluate_retrieval_gate(res)
                dist = gate.best_distance if gate.best_distance is not None else 1.0
                print(f"Query: {q:60} | Dist: {dist:.4f} | Pass: {gate.passed}")
                self.assertTrue(gate.passed, f"Valid query should pass Gate 1: {q} (Dist: {dist})")

    def test_invalid_queries(self):
        invalid_queries = [
            "Who is the current Finance Minister of India?",
            "What is the unemployment rate in India in October 2023?",
            "What did the RBI say about cryptocurrency taxation?"
        ]
        
        print("\n--- Invalid Queries ---")
        for q in invalid_queries:
            with self.subTest(query=q):
                res = search_document(q, self.doc_id)
                gate = evaluate_retrieval_gate(res)
                dist = gate.best_distance if gate.best_distance is not None else 1.0
                print(f"Query: {q:60} | Dist: {dist:.4f} | Pass: {gate.passed}")
                self.assertFalse(gate.passed, f"Invalid query should be blocked by Gate 1: {q} (Dist: {dist})")

if __name__ == "__main__":
    unittest.main()
