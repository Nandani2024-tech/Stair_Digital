import sys
import os

# Append the current directory so imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from conversation.query_rewriter import QueryRewriter
from ui.chat_panel import safe_to_reuse_context
from models import ConversationTurn
import json
import re

print("Starting Intent Routing Verification Test Cases...\n")

rewriter = QueryRewriter()

# =================
# TestCase Runner Structure
# =================
def run_test(case_name, history_str_list, new_query, mock_state):
    print(f"=== {case_name} ===")
    history = []
    
    # Build history
    for role, text in history_str_list:
        turn = ConversationTurn(role=role, content=text)
        history.append(turn)
        
    print(f"Current Context Anchor Query: '{history[-2].content if len(history) > 1 else ''}'")
    print(f"Test Query: '{new_query}'")
    
    # 1. Rewrite evaluation
    res = rewriter.rewrite(new_query, history)
    intent = getattr(res, "intent", "factual")
    anchor_used = res.anchor_used
    print(f"Rewrite -> Intent: {intent} | Anchor: {anchor_used}")
    
    is_continuation, context_reused, safe_reuse_passed = False, False, False

    # Handle Router Bounding Safely
    if intent == "followup_expand":
        if mock_state.get("last_retrievals"):
             is_continuation = context_reused = safe_reuse_passed = True
             print("[Router] Context Continutation explicitly engaged for expandable followup!")
        else:
             res.needs_clarification = True
             print("[Router] Blocked: No prior context for followup. Triggering clarification.")
    elif intent == "reasoning":
        safe_reuse_passed = safe_to_reuse_context(new_query, res.anchor_used, mock_state)
        print(f"[Router] safe_to_reuse_context evaluated: {safe_reuse_passed}")
        if safe_reuse_passed:
             is_continuation = context_reused = True
             print("[Router] Context Continutation safely mapped to prevent hallucination!")
        else:
             print("[Router] Context Divergence Trapped! Forcing fresh chromium vector extraction.")

    if res.needs_clarification:
         print("[Result] Action: CLARIFICATION UI TRIGGERED.\n")
         return
         
    if context_reused:
         print(f"[Result] Action: REUSING LAST RETRIEVALS. Skipping Gate 1 Search Block.\n")
    else:
         print(f"[Result] Action: STANDARD RETRIEVAL INITIATED for '{res.rewritten_query}'.\n")


# Case 1 (Reasoning Pass)
mock_state = {
    "last_retrievals": [{"id": 0}],
    "last_trace": {"gate1_passed": True}
}
run_test(
    case_name="Case 1 (Reasoning Fix)",
    history_str_list=[
        ("user", "Explain GDP growth."),
        ("user", "Explain GDP growth.") # Simulating UI chat_history active duplication rule internally
    ],
    new_query="Why is it important?",
    mock_state=mock_state
)

# Case 2 (Follow Up Expansion)
run_test(
    case_name="Case 2 (Follow-up Expand)",
    history_str_list=[
        ("user", "Explain inflation."),
        ("user", "Explain inflation.") 
    ],
    new_query="Explain more",
    mock_state=mock_state
)

# Case 3 (Topic Shift Trap)
run_test(
    case_name="Case 3 (Topic Shift Protection)",
    history_str_list=[
        ("user", "Explain GDP growth."),
        ("user", "Explain GDP growth.") 
    ],
    new_query="Why is it important for startups?",
    mock_state=mock_state
)

# Case 4 (No Context Expansion)
run_test(
    case_name="Case 4 (No Context Expansion Block)",
    history_str_list=[],
    new_query="Explain more",
    mock_state={"last_retrievals": []}
)
