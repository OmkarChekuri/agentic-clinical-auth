"""
AGENTIC-CLINICAL-AUTH: CORE ENGINE
RATIONALE:
1. PROMPT CACHING: System instructions are prioritized in LLM calls.
2. CONCURRENCY: Managed via a global semaphore to protect provider rate limits.
3. PERSISTENCE: Background tasks ensure reasoning loops don't block API responses.
4. A2A PROTOCOL: Structured messaging for auditability between Reviewer and Liaison agents.
"""

import os
import json
import uuid
import asyncio
import aiohttp
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# --- 1. PROVIDER LAYER (REASONING ENGINE) ---
class LLMBrain:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "ollama")
        self.model_name = os.getenv("MODEL_NAME", "llama3.2")
        self.api_key = os.getenv("GEMINI_API_KEY", "")

    async def generate(self, session: aiohttp.ClientSession, prompt: str, system_prompt: str = "", semaphore: asyncio.Semaphore = None) -> str:
        # Optimization: Prefix-cache structure
        full_context = f"CONTEXT: {system_prompt}\n\nTASK: {prompt}"
        
        async with (semaphore or asyncio.Semaphore(20)):
            try:
                if self.provider == "ollama":
                    url = f"{os.getenv('OLLAMA_HOST', 'http://localhost:11434')}/api/generate"
                    payload = {"model": self.model_name, "prompt": full_context, "stream": False}
                    async with session.post(url, json=payload, timeout=60) as resp:
                        result = await resp.json()
                        return result.get("response", "No response from local provider.")
                
                elif self.provider == "gemini":
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={self.api_key}"
                    payload = {"contents": [{"parts": [{"text": full_context}]}]}
                    async with session.post(url, json=payload, timeout=60) as resp:
                        result = await resp.json()
                        return result['candidates'][0]['content']['parts'][0]['text']
            except Exception as e:
                return f"REASONING_ERROR: {str(e)}"
        return ""

# --- 2. MULTI-SOURCE RAG (POLICY ENGINE) ---
class HealthcareRAG:
    def __init__(self):
        # In production, these are loaded from /data/policies/
        self._policies = {
            "Spinal Fusion": "POLICY SS-001: Required 6 months failed PT, BMI < 35.",
            "Knee Arthroplasty": "POLICY KS-004: Required Age > 50, KL Grade 3+.",
            "Coronary Stent": "POLICY CS-003: Required Stenosis > 70% in major vessel."
        }

    async def get_policy(self, procedure: str) -> str:
        return self._policies.get(procedure, "Standard Medical Necessity guidelines apply.")

# --- 3. MCP TOOL CLUSTER (EMR & BILLING) ---
class ClinicalMCPServer:
    async def fetch_notes(self, patient_id: str) -> str:
        # Simulation of HL7 v2 parsing
        await asyncio.sleep(0.5)
        mock_data = {
            "P123": "EMR_EXTRACT: Patient completed 6m PT. Last BMI recorded at 32.",
            "P999": "EMR_EXTRACT: Chronic knee pain. No PT history. BMI 44."
        }
        return mock_data.get(patient_id, "No clinical records found.")

class BillingMCPServer:
    async def fetch_claims(self, patient_id: str) -> List[str]:
        # Simulation of Claims DB query
        await asyncio.sleep(0.5)
        mock_claims = {"P123": ["Claim: PT_SESSIONS_1-12 (Status: Paid)"]}
        return mock_claims.get(patient_id, [])

# --- 4. MULTI-AGENT ORCHESTRATOR ---
class AgenticAuthSystem:
    def __init__(self, brain: LLMBrain, rag: HealthcareRAG, clinical: ClinicalMCPServer, billing: BillingMCPServer):
        self.brain = brain
        self.rag = rag
        self.clinical = clinical
        self.billing = billing

    async def run_reasoning_loop(self, session: aiohttp.ClientSession, case_id: str, p_id: str, proc: str, semaphore: asyncio.Semaphore):
        policy = await self.rag.get_policy(proc)
        
        # Agent Turn 1: Initial Assessment (The Reviewer)
        triage_prompt = f"Review patient {p_id} for {proc}. Policy: {policy}. Identify missing evidence."
        initial_review = await self.brain.generate(session, triage_prompt, "Role: Clinical Reviewer", semaphore)
        
        # Agent Turn 2: Data Gathering (The Liaison)
        # Triggered if Reviewer detects missing info or requires verification
        if any(keyword in initial_review.lower() for keyword in ["missing", "pt", "bmi", "history"]):
            clinical_notes = await self.clinical.fetch_notes(p_id)
            billing_history = await self.billing.fetch_claims(p_id)
            
            # Agent Turn 3: Final Audit (The Supervisor/Auditor)
            final_prompt = f"Original Case: {initial_review}\n\nFetched EMR: {clinical_notes}\nFetched Claims: {billing_history}\n\nFinal Decision?"
            return await self.brain.generate(session, final_prompt, "Role: Clinical Auditor", semaphore)
        
        return initial_review

# --- 5. FASTAPI SETUP & LIFECYCLE ---
case_db = {} # Mock DB for multi-tab polling

class AppState:
    session: Optional[aiohttp.ClientSession] = None
    semaphore: asyncio.Semaphore = asyncio.Semaphore(20)
    system: Optional[AgenticAuthSystem] = None

app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize shared resources
    app_state.session = aiohttp.ClientSession()
    app_state.system = AgenticAuthSystem(
        LLMBrain(), HealthcareRAG(), ClinicalMCPServer(), BillingMCPServer()
    )
    yield
    # Shutdown: Clean up
    await app_state.session.close()

app = FastAPI(title="Agentic Clinical Auth API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AuthRequest(BaseModel):
    patient_id: str
    procedure: str

@app.post("/api/authorize")
async def authorize(req: AuthRequest, background_tasks: BackgroundTasks):
    case_id = f"CASE-{uuid.uuid4().hex[:6].upper()}"
    case_db[case_id] = {
        "id": case_id, "patient_id": req.patient_id, "procedure": req.procedure,
        "status": "Processing", "result": None, "time": datetime.now(timezone.utc).isoformat()
    }
    background_tasks.add_task(run_background_review, case_id, req.patient_id, req.procedure)
    return {"case_id": case_id}

@app.get("/api/cases")
async def list_cases():
    return sorted(case_db.values(), key=lambda x: x['time'], reverse=True)

async def run_background_review(case_id: str, p_id: str, proc: str):
    try:
        result = await app_state.system.run_reasoning_loop(
            app_state.session, case_id, p_id, proc, app_state.semaphore
        )
        case_db[case_id]["status"] = "Completed"
        case_db[case_id]["result"] = result
    except Exception as e:
        case_db[case_id]["status"] = "Failed"
        case_db[case_id].update({"result": f"System Error: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("API_PORT", 8000)))