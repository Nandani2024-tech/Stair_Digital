from models import AgentResponse, ResponseType, Citation, ConversationTurn
import json

def trace_bug3():
    # 1. Simulate the exact text extracted previously
    clean_answer = "The repo rate was held steady at 6.50% ."
    citations = [Citation(page=2, section="RBI continues with hawkish policy stance;", chunk_id="chunk_1")]
    
    # 2. Simulate Response creation (happens in generator.py)
    response = AgentResponse(
        response_type=ResponseType.ANSWER,
        answer=clean_answer,
        citations=citations,
        gate1_passed=True,
        gate2_passed=True
    )
    
    # 3. Simulate ConversationTurn creation (happens in chat_panel.py)
    assistant_turn = ConversationTurn(
        role="assistant",
        content=response.answer,
        response_type=response.response_type,
        citations=response.citations,
        retrieved_chunks=[]
    )
    
    print("\n--- PHASE 1: AT APPEND TIME ---")
    print(f"Payload Response Content: {repr(response.answer)}")
    print(f"Payload Turn Content: {repr(assistant_turn.content)}")
    print(f"Is Content Empty?: {not bool(assistant_turn.content)}")
    
    # 4. Simulate History Render Rerun
    print("\n--- PHASE 2: AT RERENDER TIME ---")
    # Simulate st.session_state parsing back from object
    simulated_history = [assistant_turn]
    rerendered_turn = simulated_history[0]
    print(f"Rerender Turn Content: {repr(rerendered_turn.content)}")
    print(f"Is Rerender Content Empty?: {not bool(rerendered_turn.content)}")

    print("\n--- PHASE 3: CITATION HTML COLLISION TEST ---")
    # Simulate HTML render block
    html_block = " ".join(
        f'<span style="background-color: #f0f2f6; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; margin-right: 5px; border: 1px solid #dfe1e5;">'
        f'{c.render()}</span>'
        for c in rerendered_turn.citations
    )
    print("Does HTMl string overwrite Content string variable? NO.")
    print("HTML Payload:")
    print(html_block)

if __name__ == "__main__":
    trace_bug3()
