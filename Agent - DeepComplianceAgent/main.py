from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel
import os
from typing import Optional, List, Any, Dict
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
import json
import statistics
import datetime
import uuid
import tempfile
from pathlib import Path

from uipath_langchain.chat import UiPathChat
from pydantic import BaseModel

load_dotenv()
logger = logging.getLogger(__name__)

# You may keep this UiPathChat instance for other quick LLM calls, but main calls below use `call_llm`.
llm = UiPathChat(model="gpt-4o-mini-2024-07-18")




# ------------------ Helper: unified LLM caller ------------------
# Adjust SYSTEM_PROMPT single place if needed
SYSTEM_PROMPT = (
    "You are a domain expert in financial regulations and compliance analysis. "
    "Your task is to compare two versions of regulatory or policy documents, identify specific rule-level "
    "changes, produce structured findings, and explain reasoning concisely and accurately."
)

async def call_llm(prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Calls UiPathAzureChatOpenAI and returns the response content (string).
    - `system_prompt` overrides default SYSTEM_PROMPT if provided.
    """
    try:
        system_msg = system_prompt if system_prompt is not None else SYSTEM_PROMPT

        llm_client = UiPathAzureChatOpenAI(
            model="gpt-4o-2024-08-06",
            temperature=0,
            max_tokens=2048,
            timeout=None,
            max_retries=2,
        )

        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=prompt)
        ]

        response = await llm_client.ainvoke(messages)
        # response expected to have .content
        response = response.content if hasattr(response, "content") else str(response)
        logger.info(f"LLM response length: {len(response) if response else 0}")
        return response

    except Exception as e:
        logger.error(f"Failed to generate response: {str(e)}")
        print(f"Failed to generate response: {str(e)}")
        return "An error occurred while generating the response."


# ------------------ Data models ------------------
class GraphState(BaseModel):
    topic: Optional[str] = None
    # context-grounding results
    retrieved_rules: Optional[List[Any]] = None
    # LLM-produced findings
    findings: Optional[List[Any]] = None
    # anomaly detection
    anomalies: Optional[List[Any]] = None
    anomaly_score: Optional[float] = None
    # decision & actions
    decided_action: Optional[Dict[str, Any]] = None  # e.g., {"action":"escalate","reason":"...", "confidence":0.8}
    action_task_id: Optional[int] = None
    # audit / explain
    audit_id: Optional[str] = None
    audit_location: Optional[str] = None
    # optional feedback (populated externally or via learn_from_feedback)
    feedback: Optional[List[Any]] = None


class GraphOutput(BaseModel):
    report: str


class AgentState(BaseModel):
    document_path: Optional[str] = None 
    old_version_content: Optional[str] = None
    new_version_content: Optional[str] = None
    diff_output: Optional[str] = None


# ------------------ GRAPH NODES / UTILS ------------------

def read_docx_text(filepath):
    doc = Document(filepath)
    text = []
    for para in doc.paragraphs:
        if para.text.strip():
            text.append(para.text.strip())
    return "\n".join(text)


def safe_get_snippets(retrieved, n=3):
    return "\n\n".join([r.get("snippet") for r in (retrieved[:n] or []) if isinstance(r, dict) and r.get("snippet")])


# ------------------ Node: Read & Extract ------------------
async def read_and_extract(state: GraphState) -> GraphState:
    logger.info("Read & Extract: starting document read.")
    filepath = os.getenv("INPUT_FILE", "supplier.docx")
    if not os.path.exists(filepath):
        logger.warning(f"Read & Extract: file not found at {filepath}. Keeping original topic.")
        return state
    extracted_text = read_docx_text(filepath)
    logger.info(f"Read & Extract: extracted {len(extracted_text)} characters from document.")
    return GraphState(topic=extracted_text)


# ------------------ Node: Find Relevant Rules ------------------
async def find_relevant_rules(state: GraphState) -> GraphState:
    logger.info("find_relevant_rules: starting context grounding search.")
    index_name = os.getenv("CONTEXT_INDEX_NAME", "DeepComplianceIndex")
    query_text = (state.topic or "")[:4000]

    if not query_text.strip():
        logger.warning("find_relevant_rules: no document text available to query context grounding.")
        return state

    try:
        sdk = UiPath()
        results = sdk.context_grounding.search(
            name=index_name,
            query=query_text,
            number_of_results=int(os.getenv("CONTEXT_TOP_K", "5")),
            folder_path=os.getenv("FOLDER_PATH", "Shared"),
        )
        retrieved = []
        for r in results:
            snippet = getattr(r, "text", None) or getattr(r, "snippet", None) or (r.get("text") if isinstance(r, dict) else None) or str(r)
            score = getattr(r, "score", None) or (r.get("score") if isinstance(r, dict) else None)
            source = getattr(r, "source", None) or (r.get("source") if isinstance(r, dict) else None)
            retrieved.append({"snippet": snippet, "score": score, "source": source})
        return GraphState(topic=state.topic, retrieved_rules=retrieved)
    except Exception:
        logger.exception("Context grounding search failed:")
        return state
    finally:
        logger.info("find_relevant_rules: completed.")


# ------------------ Node: Reason About Compliance ------------------
async def reason_about_compliance(state: GraphState) -> GraphState:
    logger.info("reason_about_compliance: starting LLM reasoning about compliance.")
    doc_text = state.topic or ""
    retrieved = state.retrieved_rules or []

    if not doc_text.strip():
        logger.warning("reason_about_compliance: no document text available.")
        return state

    top_snips = safe_get_snippets(retrieved, n=3)
    logger.info(f"reason_about_compliance: using {len(retrieved)} retrieved rules; top snippets length: {len(top_snips)}")

    # Build the prompt for the LLM (we expect a JSON array output)
    human_prompt = (
        "Document:\n"
        f"{doc_text}\n\n"
        "Relevant policy snippets:\n"
        f"{top_snips}\n\n"
        "Task: Identify any policy violations or risks. Return a JSON array of findings. Each finding must be an object with keys: "
        "'issue' (short title), 'severity' ('High'|'Medium'|'Low'), "
        "'explanation' (one-sentence), 'confidence' (0.0-1.0 numeric), "
        "and 'evidence_refs' (list of strings identifying which policy snippet(s) support the finding). "
        "If no issues are found, return an empty JSON array: []. Provide only valid JSON as the response. Don't write anything extra as this will be passed to the json.loads method for parsing"
    )

    try:
        # call_llm returns a string (response content)
        logger.info(f"reason_about_compliance: sending prompt to LLM (length {len(human_prompt)})")
        content = await call_llm(human_prompt)
        logger.info(f"reason_about_compliance: LLM response length: {len(content) if content else 0}")
        logger.info(f"reason_about_compliance: LLM response content: {content!r}")

        # --- Robust parsing logic starts here ---

        def _strip_code_fences(s: str) -> str:
            # Remove triple-backtick code fences and common language labels like ```json
            s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)   # leading fence
            s = re.sub(r"\s*```$", "", s, flags=re.IGNORECASE)            # trailing fence
            # Also remove any standalone leading/trailing backticks just in case
            s = s.strip(" \n\r\t`")
            return s.strip()

        def _extract_first_json_array(s: str) -> Optional[str]:
            """
            Find the first balanced JSON array substring in s (starting with '[').
            Returns the substring including both brackets, or None if not found.
            This is robust to nested objects/brackets inside the array.
            """
            start = s.find("[")
            if start == -1:
                return None
            depth = 0
            for i in range(start, len(s)):
                ch = s[i]
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        return s[start:i+1]
            return None

        parsed = None
        raw = content or ""

        # 1) quick attempt: strip common fences then json.loads entire text
        candidate = _strip_code_fences(raw)
        try:
            parsed = json.loads(candidate)
        except Exception:
            # 2) if that fails, try to extract the first balanced JSON array substring and parse it
            arr_text = _extract_first_json_array(raw)
            if arr_text:
                try:
                    parsed = json.loads(arr_text)
                except Exception:
                    parsed = None
            else:
                parsed = None

        # --- Robust parsing logic ends here ---

        if parsed is None:
            logger.exception("Failed to parse LLM findings JSON after cleaning attempts; storing raw response in findings.")
            fallback = [{
                "issue": "LLM-parsing-error",
                "severity": "Medium",
                "explanation": "Could not parse LLM JSON output. See raw_output in evidence_refs.",
                "confidence": 0.0,
                "evidence_refs": [raw[:1000]]
            }]
            return GraphState(topic=state.topic, retrieved_rules=state.retrieved_rules, findings=fallback)

        # normalize to list if single object was returned
        if isinstance(parsed, dict):
            parsed = [parsed]

        return GraphState(topic=state.topic, retrieved_rules=state.retrieved_rules, findings=parsed)

    except Exception:
        logger.exception("reason_about_compliance: LLM call failed.")
        return state
    finally:
        logger.info("reason_about_compliance: completed.")



# ------------------ Node: Check for Anomalies ------------------
async def check_for_anomalies(state: GraphState) -> GraphState:
    logger.info("check_for_anomalies: starting anomaly detection.")
    doc_text = state.topic or ""
    if not doc_text.strip():
        logger.warning("check_for_anomalies: no document text available.")
        return state

    number_regex = re.compile(r"(?P<full>(?:[£$€₹])?\s?\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d+)?|\b\d+(?:[.,]\d+)?\b)")
    matches = number_regex.finditer(doc_text)
    logger.info(f"check_for_anomalies: found {len(list(number_regex.finditer(doc_text)))} numeric values in document.")

    values = []
    snippets = []
    for m in matches:
        s = m.group("full")
        cleaned = re.sub(r"[£$€₹\s]", "", s)
        cleaned = cleaned.replace(",", "")
        try:
            val = float(cleaned)
            values.append(val)
            start, end = max(0, m.start() - 30), min(len(doc_text), m.end() + 30)
            snippets.append(doc_text[start:end].replace("\n", " "))
        except Exception:
            continue

    anomalies = []
    anomaly_score = 0.0
    if values and len(values) >= 2:
        try:
            mean_v = statistics.mean(values)
            stdev_v = statistics.pstdev(values) if statistics.pstdev(values) != 0 else 1e-9
            z_scores = [ (v - mean_v) / stdev_v for v in values ]
            for v, z, sn in zip(values, z_scores, snippets):
                if abs(z) > 2.0:
                    anomalies.append({"value": v, "z_score": z, "context": sn})
            maxz = max([abs(z) for z in z_scores]) if z_scores else 0.0
            anomaly_score = min(1.0, maxz / 5.0)
        except Exception:
            logger.exception("check_for_anomalies: failed to compute stats.")
            anomalies = []
            anomaly_score = 0.0
    else:
        anomalies = []
        anomaly_score = 0.0
    
    logger.info(f"check_for_anomalies: detected {len(anomalies)} anomalies with anomaly_score={anomaly_score:.2f}")

    return GraphState(
        topic=state.topic,
        retrieved_rules=state.retrieved_rules,
        findings=state.findings,
        anomalies=anomalies,
        anomaly_score=anomaly_score,
    )


# ------------------ NEW: Decide Action ------------------
async def decide_action(state: GraphState) -> GraphState:
    logger.info("decide_action: starting decision logic.")
    findings = state.findings or []
    anomaly_score = state.anomaly_score or 0.0

    severities = [ (f.get("severity","Low") if isinstance(f, dict) else "Low") for f in findings ]
    confidences = [ (float(f.get("confidence",0.0)) if isinstance(f, dict) and f.get("confidence") is not None else 0.0) for f in findings ]
    avg_conf = sum(confidences)/len(confidences) if confidences else 0.0

    action = "log_and_monitor"
    reason = []
    combined_confidence = avg_conf * (1.0 - anomaly_score)
    logger.info(f"decide_action: severities={severities}, avg_conf={avg_conf:.2f}, anomaly_score={anomaly_score:.2f}, combined_confidence={combined_confidence:.2f}")

    if any(s.lower() == "high" for s in severities):
        action = "escalate"
        reason.append("Presence of high severity finding(s).")
    elif anomaly_score > 0.5:
        action = "escalate"
        reason.append(f"High anomaly score {anomaly_score:.2f}.")
    elif any(s.lower() == "medium" for s in severities) or anomaly_score > 0.2:
        action = "log_and_monitor"
        reason.append("Medium severity or moderate anomaly score.")
    elif all(s.lower() == "low" for s in severities) and avg_conf > 0.8 and anomaly_score < 0.2:
        action = "auto_resolve"
        reason.append("All low severity with high confidence and low anomaly score.")
    else:
        reason.append("Defaulting to log_and_monitor based on heuristics.")

    decided = {
        "action": action,
        "reason": " ".join(reason),
        "combined_confidence": min(1.0, max(0.0, combined_confidence)),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }

    logger.info(f"decide_action: decided action={action} with reason: {' '.join(reason)}")

    return GraphState(
        topic=state.topic,
        retrieved_rules=state.retrieved_rules,
        findings=state.findings,
        anomalies=state.anomalies,
        anomaly_score=state.anomaly_score,
        decided_action=decided,
    )


# ------------------ NEW: Act or Escalate ------------------
async def act_or_escalate(state: GraphState) -> GraphState:
    logger.info("act_or_escalate: starting action execution.")
    sdk = UiPath()
    decision = state.decided_action or {}
    action = decision.get("action", "log_and_monitor")

    task_id = None
    try:
        if action == "auto_resolve":
            logger.info(f"Auto-resolving with confidence {decision.get('combined_confidence')}")
            task_id = None
        elif action == "log_and_monitor":
            logger.info("Logging and setting monitor flag for this document.")
            task_id = None
        elif action == "escalate":
            try:
                # Build the payload that will be stored in the Action's `data` field
                action_data = {
                    "reason": decision.get("reason"),
                    "findings": state.findings,
                    "anomaly_score": state.anomaly_score,
                    "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                }

                logger.info(f"Creating Action Center task with data: {json.dumps(action_data)[:1000]}")

                # Optional app-specific fields (set via env if you want app-specific actions)
                app_name = os.getenv("ACTION_APP_NAME","ComplianceReviewApp")   # e.g., "ComplianceReviewApp"
                #app_key = os.getenv("ACTION_APP_KEY","")     # or app key if you prefer
                #assignee = os.getenv("ACTION_ASSIGNEE","")   # optional username/email to assign

                # Call the SDK correctly: title + data=...
                if hasattr(sdk, "actions") and callable(getattr(sdk.actions, "create", None)):
                    logger.info("Using UiPath SDK actions.create to create Action Center task.")
                    # synchronous create
                    task_obj = sdk.actions.create(
                        title= "Compliance review required-",
                        data=action_data,
                        app_name=app_name,
                        app_version=1,
                        app_folder_path="Shared",
                    )
                    # returned object may be an object or dict
                    task_id = getattr(task_obj, "id", None) or (task_obj.get("id") if isinstance(task_obj, dict) else None)
                    logger.info(f"Created Action Center task with ID: {task_id}")

                elif hasattr(sdk, "actions") and callable(getattr(sdk.actions, "create_async", None)):
                    # async create variant - call it and wait
                    # note: create_async is not an awaitable descriptor on old SDKs; check your SDK version
                    logger.info("Using UiPath SDK actions.create_async to create Action Center task.")
                    task_obj = sdk.actions.create_async(
                        title= "Compliance review required-1",
                        data=action_data,
                        app_name=app_name,
                        app_version=1,
                        app_folder_path="Shared",
                    )
                    # if create_async returns an object directly, extract id; otherwise adapt per SDK
                    task_id = getattr(task_obj, "id", None) or (task_obj.get("id") if isinstance(task_obj, dict) else None)
                    logger.info(f"Created Action Center task with ID: {task_id}")

                else:
                    logger.warning("UiPath SDK does not expose actions.create/create_async - using local escalation marker.")
                    task_id = f"local-escalation-{uuid.uuid4()}"

            except Exception as e:
                logger.exception("Failed to create Action Center task; creating local escalation marker.")
                logger.exception(str(e))
                task_id = f"escalation-failed-{uuid.uuid4()}"

        else:
            logger.warning(f"Unknown action '{action}'. Defaulting to log_and_monitor.")
            task_id = None

        return GraphState(
            topic=state.topic,
            retrieved_rules=state.retrieved_rules,
            findings=state.findings,
            anomalies=state.anomalies,
            anomaly_score=state.anomaly_score,
            decided_action=state.decided_action,
            action_task_id=task_id,
        )

    except Exception:
        logger.exception("act_or_escalate: unexpected error")
        return state
    finally:
        logger.info("act_or_escalate: completed.")  
   



# ------------------ NEW: Explain & Record (Audit Trail) ------------------
async def explain_and_record(state: GraphState) -> GraphState:
    logger.info("explain_and_record: starting audit trail recording.")
    sdk = UiPath()
    audit = {
        "audit_id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "document_excerpt": (state.topic or "")[:2000],
        "retrieved_rules": state.retrieved_rules,
        "findings": state.findings,
        "anomalies": state.anomalies,
        "anomaly_score": state.anomaly_score,
        "decision": state.decided_action,
        "action_task_id": state.action_task_id,
    }

    try:
        tf = Path(tempfile.gettempdir()) / f"audit_{audit['audit_id']}.json"
        tf.write_text(json.dumps(audit, default=str, ensure_ascii=False), encoding="utf-8")

        bucket_name = os.getenv("AUDIT_BUCKET", "DeepComplianceBucket_Log")
        blob_file_path = f"audits/{tf.name}"
        try:
            if hasattr(sdk, "buckets") and hasattr(sdk.buckets, "upload"):
                sdk.buckets.upload(
                    name=bucket_name,
                    source_path=str(tf),
                    blob_file_path=blob_file_path,
                    folder_path=os.getenv("FOLDER_PATH", "Shared"),
                )
                audit_location = f"{bucket_name}/{blob_file_path}"
            else:
                logger.warning("UiPath SDK buckets.upload not available - audit saved locally.")
                audit_location = str(tf)
        except Exception:
            logger.exception("Failed to upload audit file - keeping local path.")
            audit_location = str(tf)

        return GraphState(
            topic=state.topic,
            retrieved_rules=state.retrieved_rules,
            findings=state.findings,
            anomalies=state.anomalies,
            anomaly_score=state.anomaly_score,
            decided_action=state.decided_action,
            action_task_id=state.action_task_id,
            audit_id=audit["audit_id"],
            audit_location=audit_location,
        )
    except Exception:
        logger.exception("explain_and_record: failed to write audit")
        return state


# ------------------ OPTIONAL: Learn from Feedback ------------------
async def learn_from_feedback(state: GraphState) -> GraphState:
    sdk = UiPath()
    feedback_list = state.feedback or []
    if not feedback_list:
        return state

    try:
        fb_record = {
            "feedback_id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "feedback": feedback_list,
            "related_audit": state.audit_id,
        }
        tf = Path(tempfile.gettempdir()) / f"feedback_{fb_record['feedback_id']}.json"
        tf.write_text(json.dumps(fb_record, default=str, ensure_ascii=False), encoding="utf-8")

        bucket_name = os.getenv("FEEDBACK_BUCKET", "DeepComplianceBucket_Log")
        blob_file_path = f"feedback/{tf.name}"
        try:
            if hasattr(sdk, "buckets") and hasattr(sdk.buckets, "upload"):
                sdk.buckets.upload(
                    name=bucket_name,
                    source_path=str(tf),
                    blob_file_path=blob_file_path,
                    folder_path=os.getenv("FOLDER_PATH", None),
                )
                logger.info("Feedback uploaded to bucket.")
            else:
                logger.warning("UiPath SDK buckets.upload not available - feedback saved locally.")
        except Exception:
            logger.exception("Failed to upload feedback; saved locally.")
        return state
    except Exception:
        logger.exception("learn_from_feedback: failed to persist feedback.")
        return state


# ------------------ Generate Report (final) ------------------
async def generate_report(state: GraphState) -> GraphOutput:
    prompt_text = state.topic or ""

    if state.retrieved_rules:
        top_snips = safe_get_snippets(state.retrieved_rules, n=3)
        prompt_text = f"Document:\n{prompt_text}\n\nRelevant policy snippets:\n{top_snips}"

    if state.findings:
        try:
            findings_summary = json.dumps(state.findings[:5], default=str)
        except Exception:
            findings_summary = str(state.findings)
        prompt_text = f"{prompt_text}\n\nFindings:\n{findings_summary}"

    if state.anomalies:
        try:
            anomalies_summary = json.dumps(state.anomalies[:5], default=str)
        except Exception:
            anomalies_summary = str(state.anomalies)
        prompt_text = f"{prompt_text}\n\nAnomalies:\n{anomalies_summary}\nAnomalyScore:{state.anomaly_score}"

    if state.decided_action:
        prompt_text = f"{prompt_text}\n\nDecision:\n{json.dumps(state.decided_action, default=str)}"
        if state.action_task_id:
            prompt_text = f"{prompt_text}\nActionTaskID:{state.action_task_id}"
        if state.audit_id:
            prompt_text = f"{prompt_text}\nAuditID:{state.audit_id}"

    # Use call_llm for generating the final human-readable report
    content = await call_llm(prompt_text, system_prompt="You are a report generator. Provide a concise human-readable report based on the following information.")
    return GraphOutput(report=content if content else "")


# ------------------ Build Graph ------------------
builder = StateGraph(GraphState, output=GraphOutput)

builder.add_node("read_and_extract", read_and_extract)
builder.add_node("find_relevant_rules", find_relevant_rules)
builder.add_node("reason_about_compliance", reason_about_compliance)
builder.add_node("check_for_anomalies", check_for_anomalies)
builder.add_node("decide_action", decide_action)
builder.add_node("act_or_escalate", act_or_escalate)
builder.add_node("explain_and_record", explain_and_record)
builder.add_node("generate_report", generate_report)
# learn_from_feedback is available but not placed in the main linear flow;
# you may call it separately when feedback arrives.
builder.add_node("learn_from_feedback", learn_from_feedback)

# Wire the main linear flow:
builder.add_edge(START, "read_and_extract")
builder.add_edge("read_and_extract", "find_relevant_rules")
builder.add_edge("find_relevant_rules", "reason_about_compliance")
builder.add_edge("reason_about_compliance", "check_for_anomalies")
builder.add_edge("check_for_anomalies", "decide_action")
builder.add_edge("decide_action", "act_or_escalate")
builder.add_edge("act_or_escalate", "explain_and_record")
builder.add_edge("explain_and_record", "generate_report")
builder.add_edge("generate_report", END)

graph = builder.compile(checkpointer=MemorySaver())
