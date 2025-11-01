from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel
import os
from typing import Optional
import asyncio
from docx import Document
from uipath_langchain.chat.models import UiPathAzureChatOpenAI
import logging
from dotenv import load_dotenv
import csv
from docx.table import Table
from docx.text.paragraph import Paragraph
import re
from uipath import UiPath

load_dotenv()

logger = logging.getLogger(__name__)

class GraphInput(BaseModel):
    sdd_path: Optional[str] = None 

class GraphOutput(BaseModel):
    report: str

class AgentState(BaseModel):
    document_path: Optional[str] = None 
    old_version_content: Optional[str] = None
    new_version_content: Optional[str] = None
    diff_output: Optional[str] = None



# ------------------ UTILS ------------------
def read_docx_text(filepath):
    doc = Document(filepath)
    text = []
    for para in doc.paragraphs:
        if para.text.strip():
            text.append(para.text.strip())
    return "\n".join(text)



# ------------------ LLM ------------------

async def call_llm(prompt: str) -> str:
    try:
        llm = UiPathAzureChatOpenAI(
            model="gpt-4o-2024-08-06",
            temperature=0,
            max_tokens=2048,
            timeout=None,
            max_retries=2,
        )

        messages = [
            SystemMessage(content="You are a domain expert in financial regulations and compliance analysis. Your task is to compare two versions of regulatory or policy documents (such as Basel III vs Basel 3.1), identify specific rule-level changes, and present them in a clear and structured manner."),
            HumanMessage(content=prompt)
        ]

        response =await llm.ainvoke(messages)

        logger.info(f"LLM response: {response.content}")

        return response.content

    except Exception as e:
        logger.error(f"Failed to generate response: {str(e)}")
        print(f"Failed to generate response: {str(e)}")
        return "An error occurred while generating the response."



# ------------------ GRAPH NODES ------------------

def download_document(state: AgentState) -> AgentState:
    sdk = UiPath()
    #Download a file from a bucket
    # sdk.buckets.download(
    #     name="Documents",
    #     blob_file_path= "BASEL 3 requirements.docx",
    #     destination_path= "BASEL 3 requirements.docx",
    #     folder_path = "Shared"
    # )

    # sdk.buckets.download(
    #     name="Documents",
    #     blob_file_path= "BASEL 3-3.1 requirements.docx",
    #     destination_path= "BASEL 3-3.1 requirements.docx",
    #     folder_path = "Shared"
    # )

    return state

async def read_document_content(state: AgentState) -> AgentState:
    state.old_version_content = read_docx_text("BASEL 3 requirements.docx")
    state.new_version_content = read_docx_text("BASEL 3-3.1 requirements.docx")
    return state



async def compare_documents_with_llm(state: AgentState) -> AgentState:
    prompt = f"""
You are a domain expert in financial regulations and capital adequacy compliance.

Below are two versions of Basel capital requirement frameworks.

Your task is to:
- Identify rule-level **regulatory or structural** differences between Basel 3.1 and Basel III.
- Ignore identical rules or superficial changes (e.g., wording tweaks).
- Focus only on meaningful additions, removals, or modifications in capital requirement rules.
- Your output **must** be a markdown table with the following 5 columns:

| Rule Area | Change Type | Basel 3.1 (A) | Basel III (B) | RuleID|
|-----------|-------------|----------------|----------------|-------------------------------|

- "Change Type" must be one of: `Added`, `Removed`, or `Modified`.
- Ensure all differences are explained clearly and specifically.
- you should only return markdown nothing else

---

### Basel 3.1 (Document A)
{state.new_version_content}

---

### Basel III (Document B)
{state.old_version_content}
"""

    state.diff_output = await call_llm(prompt)
    return state

async def export_to_csv(state: AgentState) -> AgentState:
    print("Exporting LLM-generated JIRA stories to CSV...")
    
    headers, rows = parse_markdown_table(state.diff_output)

    filename = "diff_output.csv"
    output_path = os.path.join(os.getcwd(), filename)

    with open(output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"CSV file saved: {output_path}")

    sdk = UiPath()
    # upload a file from a bucket
    # sdk.buckets.upload(
    #     name = "Documents",
    #     source_path = filename,
    #     blob_file_path = filename,
    #     folder_path = "Shared",
    #     content_type = "text/csv"
    # )

    #state.jira_output_path = output_path
    return state


def parse_markdown_table(markdown_table: str):
    # Remove triple backtick code fences (if present)
    lines = markdown_table.strip().splitlines()
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]

    # Strip leading markdown language tag if present (like ```markdown)
    lines = [line.strip() for line in lines if line.strip()]
    if lines and lines[0].lower().startswith("markdown"):
        lines = lines[1:]

    # Parse headers and rows
    headers = [col.strip() for col in lines[0].split('|') if col.strip()]
    rows = []
    for line in lines[2:]:  # Skip header and separator
        if '|' not in line:
            continue
        row = [col.strip() for col in line.split('|') if col.strip()]
        if len(row) == len(headers):
            rows.append(row)
    return headers, rows



# ------------------ GRAPH BUILD ------------------

builder = StateGraph(AgentState, input_schema=GraphInput , output_schema=GraphOutput)
builder.add_node("download_document",download_document)
builder.add_node("read_document_content", read_document_content)
builder.add_node("compare_documents_with_llm", compare_documents_with_llm)
builder.add_node("export_to_csv",export_to_csv)

builder.set_entry_point("download_document")
builder.add_edge("download_document", "read_document_content")
builder.add_edge("read_document_content", "compare_documents_with_llm")
builder.add_edge("compare_documents_with_llm", "export_to_csv")
builder.set_finish_point("export_to_csv")

graph = builder.compile(checkpointer=MemorySaver())

