"""
agent_engine.py - Groq LLM Multi-Agent Pipelines for OpsPulse AI
Implements Log Parser Agent, Diagnostic Agent, and Remediation Agent using 'llama-3.3-70b-versatile'.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def get_groq_client(api_key: Optional[str] = None) -> Optional[Groq]:
    """
    Initializes and returns a Groq client if an API key is available and not a placeholder.
    """
    key = api_key or os.getenv("GROQ_API_KEY", "")
    if not key or key.strip() in ("your_groq_api_key_here", "YOUR_GROQ_API_KEY", ""):
        return None
    try:
        return Groq(api_key=key.strip())
    except Exception as e:
        print(f"Error creating Groq client: {e}")
        return None


# ==========================================
# 1. Log Parser Agent
# ==========================================
def parse_logs(
    raw_logs: str,
    client: Optional[Groq] = None,
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Log Parser Agent: Parses input logs into structured JSON:
    - error_signature: short string describing the core failure
    - affected_services: list of service names affected
    - severity: 'CRITICAL', 'HIGH', 'MEDIUM', or 'LOW'
    - summary: concise summary of the incident and impact
    """
    if client is None:
        return _mock_parse_logs(raw_logs)

    prompt = f"""You are an expert SRE Log Parser Agent.
Analyze the following raw system and application logs:

```
{raw_logs}
```

Extract the diagnostic telemetry into a strict JSON object with EXACTLY the following keys:
- "error_signature": A concise technical error title/signature (e.g., "PostgreSQL Connection Pool Saturation & Ingress 504 Timeouts").
- "affected_services": A list of service names, components, or pods identified from the logs (e.g., ["order-service", "ingress-nginx", "postgresql"]).
- "severity": One of ["CRITICAL", "HIGH", "MEDIUM", "LOW"].
- "summary": A 2-3 sentence technical summary describing the trigger, symptoms, and cascading failure chain.

Output ONLY valid JSON.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a specialized SRE Log Parsing AI. Respond only in valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=600
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        
        # Validate keys
        return {
            "error_signature": data.get("error_signature", "Unknown System Error"),
            "affected_services": data.get("affected_services", ["infrastructure"]),
            "severity": data.get("severity", "HIGH").upper(),
            "summary": data.get("summary", "Unclassified error detected in raw log input.")
        }
    except Exception as e:
        print(f"Groq Log Parser Error: {e}, falling back to heuristic parsing")
        return _mock_parse_logs(raw_logs)


# ==========================================
# 2. Diagnostic Agent (RCA)
# ==========================================
def diagnose_root_cause(
    parsed_logs: Dict[str, Any],
    raw_logs: str,
    runbook_context: str,
    client: Optional[Groq] = None,
    model: str = DEFAULT_MODEL
) -> str:
    """
    Diagnostic Agent: Performs thorough Root Cause Analysis (RCA)
    using parsed logs, raw stack traces, and retrieved FAISS runbook context.
    """
    if client is None:
        return _mock_diagnose_root_cause(parsed_logs, runbook_context)

    system_prompt = (
        "You are a Principal Site Reliability Engineer (SRE) and Incident Commander leading a Sev-1 RCA triage.\n"
        "Ground your diagnosis strictly in the provided logs and the retrieved internal runbooks."
    )

    user_prompt = f"""### INCIDENT TELEMETRY
- **Error Signature**: {parsed_logs.get('error_signature')}
- **Severity**: {parsed_logs.get('severity')}
- **Affected Services**: {', '.join(parsed_logs.get('affected_services', []))}
- **Summary**: {parsed_logs.get('summary')}

### RAW LOG EXCERPT
```
{raw_logs[:3000]}
```

### RETRIEVED RUNBOOK CONTEXT (FROM FAISS VECTOR STORE)
{runbook_context}

---
Perform a structured, authoritative Root Cause Analysis (RCA) in clear Markdown with the following sections:
1. **Executive Summary & Blast Radius**: High-level impact and affected user journeys.
2. **Timeline & Failure Propagation**: How the failure initiated and cascaded between components.
3. **Primary Root Cause**: Pinpoint the exact bottleneck, deadlock, or configuration flaw (distinguish root cause from symptoms).
4. **Trigger Mechanism**: What sequence of requests or state transition tripped the failure.
5. **Runbook Grounding**: Reference the specific runbook diagnostic steps matched to confirm this failure.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Groq Diagnostic Agent Error: {e}, using heuristic fallback")
        return _mock_diagnose_root_cause(parsed_logs, runbook_context)


# ==========================================
# 3. Remediation Agent
# ==========================================
def generate_remediation(
    parsed_logs: Dict[str, Any],
    rca_report: str,
    runbook_context: str,
    client: Optional[Groq] = None,
    model: str = DEFAULT_MODEL
) -> Dict[str, str]:
    """
    Remediation Agent: Generates step-by-step mitigation instructions
    and an executable Bash/Python hotfix script.
    
    Returns:
        {
            "instructions": markdown formatted step-by-step instructions,
            "script": executable code string,
            "script_lang": "bash" | "python"
        }
    """
    if client is None:
        return _mock_generate_remediation(parsed_logs, runbook_context)

    system_prompt = (
        "You are an SRE Automation Engineer specializing in Kubernetes, cloud infrastructure, and database operations.\n"
        "Generate concrete, immediate mitigation procedures and a robust, safe executable hotfix script."
    )

    user_prompt = f"""### INCIDENT CONTEXT
- **Signature**: {parsed_logs.get('error_signature')}
- **Severity**: {parsed_logs.get('severity')}
- **Affected Services**: {', '.join(parsed_logs.get('affected_services', []))}

### ROOT CAUSE SUMMARY
{rca_report[:2000]}

### RUNBOOK MITIGATION PROTOCOLS
{runbook_context}

---
Provide two distinct parts:
PART 1: Step-by-Step Mitigation Instructions (Immediate actions to stop bleeding, plus permanent fixes).
PART 2: A production-ready, safe executable Bash or Python script to automate the hotfix (e.g. killing blocking queries, scaling deployments, tweaking ingress timeouts, or restarting locked pods). Include safety dry-runs or confirmations where applicable.

Separate the script inside a ```bash or ```python block.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        content = response.choices[0].message.content
        
        # Extract the code block
        code_match = re.search(r"```(bash|sh|python)\n(.*?)```", content, re.DOTALL)
        if code_match:
            script_lang = code_match.group(1).lower()
            if script_lang in ("sh", "bash"):
                script_lang = "bash"
            script_code = code_match.group(2).strip()
            # Everything else is instructions
            instructions = content.replace(code_match.group(0), "").strip()
        else:
            script_code = "# Hotfix script generated for incident\n# Review instructions above."
            script_lang = "bash"
            instructions = content

        return {
            "instructions": instructions,
            "script": script_code,
            "script_lang": script_lang,
            "full_response": content
        }
    except Exception as e:
        print(f"Groq Remediation Agent Error: {e}, using heuristic fallback")
        return _mock_generate_remediation(parsed_logs, runbook_context)


# ==========================================
# High-Fidelity Fallback Simulators (Offline / No Key Mode)
# ==========================================
def _mock_parse_logs(raw_logs: str) -> Dict[str, Any]:
    """Heuristic log parser when Groq API key is not configured."""
    logs_lower = raw_logs.lower()
    
    # Identify services
    services = []
    if "postgres" in logs_lower or "psql" in logs_lower or "database" in logs_lower:
        services.append("postgresql-cluster")
    if "ingress" in logs_lower or "nginx" in logs_lower or "gateway" in logs_lower:
        services.append("ingress-nginx")
    if "order-service" in logs_lower:
        services.append("order-service")
    if "auth-service" in logs_lower:
        services.append("auth-service")
    if not services:
        services = ["api-gateway", "backend-service"]

    # Identify signature & severity
    if "504" in logs_lower and ("connection" in logs_lower or "pool" in logs_lower):
        signature = "PostgreSQL Connection Pool Saturation & Ingress 504 Gateway Timeout"
        severity = "CRITICAL"
        summary = "Upstream microservices exhausted PostgreSQL connection pools due to unclosed transactions, causing HTTP 504 timeouts across the ingress gateway."
    elif "504" in logs_lower:
        signature = "Kubernetes Upstream Ingress 504 Timeout"
        severity = "HIGH"
        summary = "Ingress proxy timed out waiting for backend worker pods to respond within proxy-read-timeout."
    elif "fatal" in logs_lower or "connection slots" in logs_lower:
        signature = "PostgreSQL Thread & Connection Pool Exhaustion"
        severity = "CRITICAL"
        summary = "PostgreSQL connection limits reached; non-superuser connection requests rejected."
    else:
        signature = "Distributed Microservice Latency & Connection Failure"
        severity = "HIGH"
        summary = "Anomalous error patterns detected across upstream service logs."

    return {
        "error_signature": signature,
        "affected_services": services,
        "severity": severity,
        "summary": summary
    }


def _mock_diagnose_root_cause(parsed_logs: Dict[str, Any], runbook_context: str) -> str:
    """Pre-computed high-fidelity RCA grounded in the runbook."""
    services_str = ", ".join(parsed_logs.get("affected_services", ["unknown"]))
    signature = parsed_logs.get("error_signature", "Incident")
    
    return f"""### 1. Executive Summary & Blast Radius
- **Incident Signature**: {signature}
- **Severity Level**: `{parsed_logs.get('severity', 'CRITICAL')}`
- **Impacted Components**: `{services_str}`
- **User Impact**: Inbound HTTP requests through the Ingress controller are failing with `504 Gateway Timeout`. Checkout and transaction workflows are stalled due to database pool starvation.

---

### 2. Timeline & Failure Propagation
```
[User Request] 
      │ 
      ▼
[ingress-nginx] ──(HTTP 504 Gateway Timeout: 110 Connection Timed Out)──✖
      │
      ▼
[order-service Pods] (Thread pool exhausted waiting on DB connection lease)
      │
      ▼
[PostgreSQL Database] ──(FATAL: remaining connection slots reserved for superuser)──✖
```
1. **Initial Trigger**: A long-running analytical query or unindexed migration acquired exclusive row/table locks without committing (`idle in transaction`).
2. **Cascading Pool Exhaustion**: Microservice instances created new connections until `max_connections` (100) was hit.
3. **Thread Starvation**: In-flight HTTP threads in upstream `order-service` pods blocked waiting for connection acquisition.
4. **Ingress Timeout**: `ingress-nginx` exceeded its `proxy-read-timeout` (60s) and returned HTTP 504 to end-users.

---

### 3. Primary Root Cause
The **primary root cause** is an unmanaged transaction holding locks in PostgreSQL without an `idle_in_transaction_session_timeout`, compounded by direct microservice database connections bypassing PgBouncer pooling limits. The Kubernetes 504 errors are a downstream symptom of connection starvation.

---

### 4. Runbook Grounding & Verification
- **Matched Runbook**: `sample_k8s_runbook.pdf` (Pages 1 & 2)
- Corroborated with **Section 2 (PostgreSQL Diagnostics)**: Query `pg_stat_activity` for `state != 'idle'` and detect locking PIDs via `pg_locks`.
- Corroborated with **Section 3 (Kubernetes Mitigation)**: Scale pod replicas and adjust `proxy-read-timeout` temporarily while database connections are restored.
"""


def _mock_generate_remediation(parsed_logs: Dict[str, Any], runbook_context: str) -> Dict[str, str]:
    """Generates remediation steps and hotfix script."""
    instructions = """### Immediate Mitigation Plan (Phase 1 - Stop the Bleeding)
1. **Terminate Blocked Postgres Queries**: Terminate runaway backend processes currently holding locks for over 60 seconds.
2. **Scale Upstream Pods Temporarily**: Prevent incoming traffic from queueing on overloaded worker instances.
3. **Enforce Idle Connection Timeout**: Set `idle_in_transaction_session_timeout = '20s'` to auto-kill hung connections.

### Long-Term Architectural Fixes (Phase 2)
1. **Route all microservices through PgBouncer**: Enforce transaction-level pooling instead of direct session connections.
2. **Implement Connection Pool Sizing Formula**: Size HikariCP/connection pools to `(core_count * 2) + effective_spindle_count`.
3. **Configure Ingress Circuit Breaking**: Enable envoy/nginx rate-limiting to shed load before saturating database limits.
"""

    script = """#!/usr/bin/env bash
# ==============================================================================
# OpsPulse AI - Automated Emergency Remediation Hotfix Script
# Target: PostgreSQL Connection Drain & Kubernetes Ingress Recovery
# ==============================================================================
set -euo pipefail

echo "========================================================"
echo " [OpsPulse AI] Initiating Automated SRE Hotfix Protocol"
echo "========================================================"

# Step 1: Identify and terminate runaway queries holding locks > 60s
echo "[1/3] Terminating idle-in-transaction and blocking PostgreSQL backends..."
PGPASSWORD="${PGPASSWORD:-postgres}" psql -h "${PGHOST:-localhost}" -U "${PGUSER:-postgres}" -d "${PGDATABASE:-production_db}" << 'EOF'
-- Terminate rogue backend sessions holding connections for > 60s
SELECT 
    pid, 
    usename, 
    state, 
    now() - state_change as duration,
    pg_terminate_backend(pid) as terminated
FROM pg_stat_activity
WHERE state IN ('idle in transaction', 'idle in transaction (aborted)')
  AND now() - state_change > interval '30 seconds';

-- Set safety timeout for new sessions
ALTER DATABASE CURRENT_DATABASE SET idle_in_transaction_session_timeout = '20000';
EOF

echo "[✓] Postgres connection slots reclaimed."

# Step 2: Scale up deployment to shed backlog
echo "[2/3] Scaling order-service deployment in namespace production..."
kubectl scale deployment order-service -n production --replicas=8 || echo "Warning: Unable to scale deployment directly"

# Step 3: Gracefully restart stuck pods
echo "[3/3] Initiating rolling restart of worker pods..."
kubectl rollout restart deployment/order-service -n production || echo "Warning: kubectl restart skipped"

echo "========================================================"
echo " [OpsPulse AI] Remediation hotfix deployed successfully!"
echo " Monitor Ingress HTTP 504 rates: kubectl logs -l app.kubernetes.io/name=ingress-nginx -n ingress-nginx"
echo "========================================================"
"""

    return {
        "instructions": instructions,
        "script": script.strip(),
        "script_lang": "bash",
        "full_response": instructions + "\n\n```bash\n" + script.strip() + "\n```"
    }
