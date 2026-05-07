import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "student_analysis", ".env"), override=True)

from langchain_upstage import ChatUpstage

try:
    llm = ChatUpstage(model="solar-pro3")
    test_response = llm.invoke("Hello")
    print(f"Test response: {test_response.content[:50]}...")
except Exception as e:
    print(f"Error: {e}")