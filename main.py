import os
import re
import json
import base64
import stat
import shutil
import asyncio
import logging
import sys
import time
from typing import List, Optional
from datetime import datetime
import httpx
import git
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GEMINI_API_KEY: str = Field("", env="GEMINI_API_KEY")
    GITHUB_TOKEN: str = Field("", env="GITHUB_TOKEN")
    GITHUB_USERNAME: str = Field("", env="GITHUB_USERNAME")
    STUDENT_SECRET: str = Field("", env="STUDENT_SECRET")
    LOG_FILE_PATH: str = Field("logs/app.log", env="LOG_FILE_PATH")
    MAX_CONCURRENT_TASKS: int = Field(2, env="MAX_CONCURRENT_TASKS")
    KEEP_ALIVE_INTERVAL_SECONDS: int = Field(30, env="KEEP_ALIVE_INTERVAL_SECONDS")
    GITHUB_API_BASE: str = Field("https://api.github.com", env="GITHUB_API_BASE")
    GITHUB_PAGES_BASE: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
settings = Settings()
if not settings.GITHUB_PAGES_BASE:
    settings.GITHUB_PAGES_BASE = f"https://{settings.GITHUB_USERNAME}.github.io"

os.makedirs(os.path.dirname(settings.LOG_FILE_PATH), exist_ok=True)
logger = logging.getLogger("task_receiver")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
file_handler = logging.FileHandler(settings.LOG_FILE_PATH, mode="a", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.handlers = []
logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.propagate = False

def flush_logs():
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        for h in logger.handlers:
            try:
                h.flush()
            except Exception:
                pass
    except Exception:
        pass

class Attachment(BaseModel):
    name: str
    url: str
class TaskRequest(BaseModel):
    task: str
    email: str
    round: int
    brief: str
    evaluation_url: str
    nonce: str
    secret: str
    attachments: List[Attachment] = []

app = FastAPI(
    title="Automated Task Receiver & Processor (Robust)",
    description="Receive tasks, generate code via LLM, deploy to GitHub Pages, and notify evaluator."
)
background_tasks_list: List[asyncio.Task] = []
task_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_TASKS)

last_received_task: Optional[dict] = None

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
GEMINI_API_KEY = settings.GEMINI_API_KEY

def verify_secret(secret_from_request: str) -> bool:
    return secret_from_request == settings.STUDENT_SECRET

def is_image_data_uri(data_uri: str) -> bool:
    if not data_uri or not data_uri.startswith("data:"):
        return False
    return re.search(r"data:image/[^;]+;base64,", data_uri, re.IGNORECASE) is not None

def data_uri_to_gemini_part(data_uri: str) -> Optional[dict]:
    if not data_uri or not data_uri.startswith("data:"):
        logger.warning("Invalid data URI provided to data_uri_to_gemini_part.")
        return None
    match = re.search(r"data:(?P<mime_type>[^;]+);base64,(?P<base64_data>.*)", data_uri, re.IGNORECASE)
    if not match:
        logger.warning("Could not parse MIME type or base64 data from URI.")
        return None
    mime_type = match.group("mime_type")
    base64_data = match.group("base64_data")
    if not mime_type.startswith("image/"):
        logger.info(f"Skipping non-image MIME type: {mime_type}")
        return None
    return {"inlineData": {"data": base64_data, "mimeType": mime_type}}

def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)

async def save_generated_files_locally(task_id: str, files: dict) -> str:
    base_dir = os.path.join(os.getcwd(), "generated_tasks")
    task_dir = os.path.join(base_dir, task_id)
    safe_makedirs(task_dir)
    logger.info(f"[LOCAL_SAVE] Saving generated files to: {task_dir}")

    for filename, content in files.items():
        file_path = os.path.join(task_dir, filename)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"   -> Saved: {filename} (bytes: {len(content)})")
        except Exception as e:
            logger.exception(f"Failed to save generated file {filename}: {e}")
            raise
    flush_logs()
    return task_dir

async def save_attachments_locally(task_dir: str, attachments: List[Attachment]) -> List[str]:
    saved_files = []
    logger.info(f"[ATTACHMENTS] Processing {len(attachments)} attachments for {task_dir}")
    for attachment in attachments:
        filename = attachment.name
        data_uri = attachment.url
        if not filename or not data_uri or not data_uri.startswith("data:"):
            logger.warning(f"Skipping invalid attachment: {filename}")
            continue
        match = re.search(r"base64,(.*)", data_uri, re.IGNORECASE)
        if not match:
            logger.warning(f"No base64 content found for attachment: {filename}")
            continue
        base64_data = match.group(1)
        file_path = os.path.join(task_dir, filename)
        try:
            file_bytes = base64.b64decode(base64_data)
            with open(file_path, "wb") as f:
                f.write(file_bytes)
            logger.info(f"   -> Saved Attachment: {filename} (bytes: {len(file_bytes)})")
            saved_files.append(filename)
        except Exception as e:
            logger.exception(f"Failed to save attachment {filename}: {e}")
            raise
    flush_logs()
    return saved_files

def remove_local_path(path: str):
    if not os.path.exists(path):
        return
    def onerror(func, path_arg, exc_info):
        try:
            os.chmod(path_arg, stat.S_IWUSR)
            func(path_arg)
        except Exception as exc:
            logger.exception(f"Failed in rmtree on {path_arg}: {exc}")
            raise
    logger.info(f"[CLEANUP] Removing local directory: {path}")
    shutil.rmtree(path, onerror=onerror)
    flush_logs()

async def setup_local_repo(local_path: str, repo_name: str, repo_url_auth: str, repo_url_http: str, round_index: int) -> git.Repo:
    github_token = settings.GITHUB_TOKEN
    github_username = settings.GITHUB_USERNAME
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    async with httpx.AsyncClient(timeout=45) as client:
        try:
            if round_index == 1:
                logger.info(f"R1: Creating remote repository '{repo_name}'...")
                payload = {"name": repo_name, "private": False, "auto_init": True}
                response = await client.post(f"{settings.GITHUB_API_BASE}/user/repos", json=payload, headers=headers)
                response.raise_for_status()
                repo = git.Repo.init(local_path)
                repo.create_remote('origin', repo_url_auth)
                logger.info("R1: Local git repository initialized.")
            else:
                logger.info(f"R{round_index}: Cloning repository from {repo_url_http}")
                repo = git.Repo.clone_from(repo_url_auth, local_path)
                logger.info(f"R{round_index}: Repository cloned.")
            flush_logs()
            return repo
        except httpx.HTTPStatusError as e:
            logger.exception(f"GitHub API call failed during repository setup: {e.response.status_code} {e.response.text}")
            raise
        except git.GitCommandError as e:
            logger.exception(f"Git command failed during setup: {e}")
            raise

async def commit_and_publish(repo: git.Repo, task_id: str, round_index: int, repo_name: str) -> dict:
    github_username = settings.GITHUB_USERNAME
    github_token = settings.GITHUB_TOKEN
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    repo_url_http = f"https://github.com/{github_username}/{repo_name}"

    async with httpx.AsyncClient(timeout=45) as client:
        try:
            repo.git.add(A=True)
            commit_message = f"Task {task_id} - Round {round_index}: automated update"
            repo.index.commit(commit_message)
            commit_sha = repo.head.object.hexsha
            logger.info(f"Committed changes, SHA: {commit_sha}")
            repo.git.branch('-M', 'main')
            repo.git.push('--set-upstream', 'origin', 'main', force=True)
            logger.info("Pushed changes to origin/main")
            await asyncio.sleep(2)
            pages_api_url = f"{settings.GITHUB_API_BASE}/repos/{github_username}/{repo_name}/pages"
            pages_payload = {"source": {"branch": "main", "path": "/"}}
            pages_max_retries = 5
            pages_base_delay = 3
            for attempt in range(pages_max_retries):
                try:
                    pages_response = await client.get(pages_api_url, headers=headers)
                    is_configured = (pages_response.status_code == 200)
                    if is_configured:
                        logger.info(f"Pages exists. Updating configuration (attempt {attempt+1})")
                        await client.put(pages_api_url, json=pages_payload, headers=headers)
                    else:
                        logger.info(f"Creating Pages config (attempt {attempt+1})")
                        await client.post(pages_api_url, json=pages_payload, headers=headers)
                    logger.info("Pages configuration succeeded.")
                    break
                except httpx.HTTPStatusError as e:
                    text = e.response.text.lower() if e.response and e.response.text else ""
                    if e.response.status_code == 422 and "main branch must exist" in text and attempt < pages_max_retries - 1:
                        delay = pages_base_delay * (2 ** attempt)
                        logger.warning(f"Timing issue configuring pages, retrying in {delay}s")
                        await asyncio.sleep(delay)
                        continue
                    logger.exception(f"Failed to configure GitHub Pages: {e.response.status_code} {e.response.text}")
                    raise
            await asyncio.sleep(5)
            pages_url = f"{settings.GITHUB_PAGES_BASE}/{repo_name}/"
            flush_logs()
            return {"repo_url": repo_url_http, "commit_sha": commit_sha, "pages_url": pages_url}
        except git.GitCommandError as e:
            logger.exception("Git operation failed during deployment.")
            raise
        except httpx.HTTPStatusError as e:
            logger.exception("GitHub API error during deployment.")
            raise

async def call_llm_for_code(prompt: str, task_id: str, image_parts: list) -> dict:
    logger.info(f"[LLM_CALL] Generating code for task {task_id}")
    system_prompt = (
        "You are an expert full-stack engineer. Produce a JSON object with keys "
        "'index.html', 'README.md', and 'LICENSE'. 'index.html' must be a single-file "
        "responsive HTML using Tailwind CSS. 'LICENSE' must be full MIT license text."
    )
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "index.html": {"type": "STRING"},
            "README.md": {"type": "STRING"},
            "LICENSE": {"type": "STRING"}
        },
        "required": ["index.html", "README.md", "LICENSE"]
    }
    contents = []
    if image_parts:
        contents.append({"parts": image_parts + [{"text": prompt}]})
    else:
        contents.append({"parts": [{"text": prompt}]})
    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        }
    }
    max_retries = 3
    base_delay = 1
    for attempt in range(max_retries):
        try:
            if not GEMINI_API_KEY:
                raise Exception("GEMINI_API_KEY not configured in environment.")
            url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
                resp.raise_for_status()
                result = resp.json()
                candidates = result.get("candidates", [])
                if not candidates:
                    raise ValueError("No candidates in LLM response")
                content_parts = candidates[0].get("content", {}).get("parts", [])
                if not content_parts:
                    raise ValueError("No content parts in LLM candidate")
                json_text = content_parts[0].get("text")
                generated_files = json.loads(json_text)
                logger.info(f"[LLM_CALL] Successfully generated files on attempt {attempt+1}")
                flush_logs()
                return generated_files
        except httpx.HTTPStatusError as e:
            logger.warning(f"[LLM_CALL] HTTP error attempt {attempt+1}: {e}")
        except (httpx.RequestError, KeyError, json.JSONDecodeError, ValueError) as e:
            logger.warning(f"[LLM_CALL] Processing error attempt {attempt+1}: {e}")
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)
            logger.info(f"[LLM_CALL] Retrying in {delay}s...")
            await asyncio.sleep(delay)
    logger.error("[LLM_CALL] Exhausted retries without successful generation.")
    raise Exception("LLM generation failed after retries")

async def notify_evaluation_server(evaluation_url: str, email: str, task_id: str, round_index: int, nonce: str, repo_url: str, commit_sha: str, pages_url: str) -> bool:
    payload = {
        "email": email,
        "task": task_id,
        "round": round_index,
        "nonce": nonce,
        "repo_url": repo_url,
        "commit_sha": commit_sha,
        "pages_url": pages_url
    }
    max_retries = 3
    base_delay = 1
    logger.info(f"[NOTIFY] Notifying evaluation server at {evaluation_url}")
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(evaluation_url, json=payload)
                resp.raise_for_status()
                logger.info(f"[NOTIFY] Notification succeeded: {resp.status_code}")
                flush_logs()
                return True
        except httpx.HTTPStatusError as e:
            logger.warning(f"[NOTIFY] HTTP error attempt {attempt+1}: {e}")
        except httpx.RequestError as e:
            logger.warning(f"[NOTIFY] Request error attempt {attempt+1}: {e}")
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)
            logger.info(f"[NOTIFY] Retrying in {delay}s")
            await asyncio.sleep(delay)
    logger.error("[NOTIFY] Failed to notify evaluation server after retries.")
    flush_logs()
    return False

async def generate_files_and_deploy(task_data: TaskRequest):
    acquired = False
    try:
        await task_semaphore.acquire()
        acquired = True
        logger.info(f"[PROCESS START] Task: {task_data.task} Round: {task_data.round}")
        flush_logs()

        task_id = task_data.task
        email = task_data.email
        round_index = task_data.round
        brief = task_data.brief
        evaluation_url = task_data.evaluation_url
        nonce = task_data.nonce
        attachments = task_data.attachments or []
        repo_name = task_id.replace(" ", "-").lower()
        github_username = settings.GITHUB_USERNAME
        github_token = settings.GITHUB_TOKEN
        repo_url_auth = f"https://{github_username}:{github_token}@github.com/{github_username}/{repo_name}.git"
        repo_url_http = f"https://github.com/{github_username}/{repo_name}"
        base_dir = os.path.join(os.getcwd(), "generated_tasks")
        local_path = os.path.join(base_dir, task_id)
        if os.path.exists(local_path):
            try:
                remove_local_path(local_path)
            except Exception as e:
                logger.exception(f"Cleanup failed for {local_path}: {e}")
                raise
        safe_makedirs(local_path)
        repo = await setup_local_repo(
            local_path=local_path,
            repo_name=repo_name,
            repo_url_auth=repo_url_auth,
            repo_url_http=repo_url_http,
            round_index=round_index
        )
        existing_index_html_content = None
        if round_index > 1:
            index_path = os.path.join(local_path, "index.html")
            try:
                if os.path.exists(index_path):
                    with open(index_path, "r", encoding="utf-8") as f:
                        existing_index_html_content = f.read()
                    logger.info(f"[PROMPT_CONTEXT] Successfully read existing index.html for R{round_index}.")
                else:
                    logger.warning(f"[PROMPT_CONTEXT] index.html not found in cloned repo for R{round_index}. Treating as fresh generation.")
            except Exception as e:
                logger.exception(f"[PROMPT_CONTEXT] Failed to read existing index.html: {e}. Treating as fresh generation.")
        image_parts = []
        attachment_list_for_prompt = []
        for attachment in attachments:
            if is_image_data_uri(attachment.url):
                part = data_uri_to_gemini_part(attachment.url)
                if part:
                    image_parts.append(part)
            attachment_list_for_prompt.append(attachment.name)
        logger.info(f"[LLM_INPUT] Image parts: {len(image_parts)} Attachments: {len(attachment_list_for_prompt)}")
        if round_index > 1:
            llm_prompt = (
                f"UPDATE INSTRUCTION (ROUND {round_index}): Modify the EXISTING index.html code "
                f"provided below to implement this new brief: '{brief}'. "
                "Provide FULL replacement content for the index.html file, as well as the full README.md and LICENSE. "
                "The index.html must remain a single, responsive Tailwind HTML file."
            )
            if existing_index_html_content:
                llm_prompt += (
                    "\n\n--- EXISTING index.html CONTENT TO BE MODIFIED ---\n"
                    f"{existing_index_html_content}"
                    "\n--- END EXISTING index.html CONTENT ---"
                )
            else:
                llm_prompt += " (Note: No existing index.html found, generate a new one based on the brief)."
        else:
            llm_prompt = (
                f"Generate a complete, single-file HTML web app to achieve: {brief}. "
                "Ensure single-file responsive Tailwind index.html, README.md, and MIT LICENSE."
            )
        if attachment_list_for_prompt:
            llm_prompt += f" Additional files in project root: {', '.join(attachment_list_for_prompt)}"
        generated_files = await call_llm_for_code(llm_prompt, task_id, image_parts)
        await save_generated_files_locally(task_id, generated_files)
        await save_attachments_locally(local_path, attachments)
        deployment_info = await commit_and_publish(
            repo=repo,
            task_id=task_id,
            round_index=round_index,
            repo_name=repo_name
        )
        repo_url = deployment_info["repo_url"]
        commit_sha = deployment_info["commit_sha"]
        pages_url = deployment_info["pages_url"]
        logger.info(f"[DEPLOYMENT] Success. Repo: {repo_url} Pages: {pages_url}")
        await notify_evaluation_server(
            evaluation_url=evaluation_url,
            email=email,
            task_id=task_id,
            round_index=round_index,
            nonce=nonce,
            repo_url=repo_url,
            commit_sha=commit_sha,
            pages_url=pages_url
        )
    except Exception as exc:
        logger.exception(f"[CRITICAL FAILURE] Task {task_data.task} failed: {exc}")
    finally:
        if acquired:
            task_semaphore.release()
        flush_logs()
        logger.info(f"[PROCESS END] Task: {task_data.task} Round: {task_data.round}")

def _task_done_callback(task: asyncio.Task):
    try:
        exc = task.exception()
        if exc:
            logger.error(f"[BACKGROUND TASK] Task finished with exception: {exc}")
            logger.exception(exc)
        else:
            logger.info("[BACKGROUND TASK] Task finished successfully.")
    except asyncio.CancelledError:
        logger.warning("[BACKGROUND TASK] Task was cancelled.")
    finally:
        flush_logs()

@app.post("/ready", status_code=200)
async def receive_task(task_data: TaskRequest, request: Request):
    global last_received_task, background_tasks_list
    if not verify_secret(task_data.secret):
        logger.warning(f"Unauthorized attempt for task {task_data.task} from {request.client.host if request.client else 'unknown'}")
        raise HTTPException(status_code=401, detail="Unauthorized: Secret mismatch")
    last_received_task = {
        "task": task_data.task,
        "email": task_data.email,
        "round": task_data.round,
        "brief": (task_data.brief[:250] + "...") if len(task_data.brief) > 250 else task_data.brief,
        "time": datetime.utcnow().isoformat() + "Z"
    }
    bg_task = asyncio.create_task(generate_files_and_deploy(task_data))
    bg_task.add_done_callback(_task_done_callback)
    background_tasks_list.append(bg_task)

    logger.info(f"Received task {task_data.task}. Background processing started.")
    flush_logs()

    return JSONResponse(status_code=200, content={"status": "ready", "message": f"Task {task_data.task} received and processing started."})

@app.get("/")
async def root():
    return {"message": "Task Receiver Service running. POST /ready to submit."}

@app.get("/status")
async def get_status():
    if last_received_task:
        return {"last_received_task": last_received_task, "running_background_tasks": len([t for t in background_tasks_list if not t.done()])}
    return {"message": "Awaiting first task submission to /ready"}

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat() + "Z"}

@app.get("/logs")
async def get_logs(lines: int = Query(200, ge=1, le=5000)):
    path = settings.LOG_FILE_PATH
    if not os.path.exists(path):
        return PlainTextResponse("Log file not found.", status_code=404)
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            buffer = bytearray()
            block_size = 1024
            blocks = 0
            while file_size > 0 and len(buffer) < lines * 2000 and blocks < 1024:
                read_size = min(block_size, file_size)
                f.seek(file_size - read_size)
                buffer.extend(f.read(read_size))
                file_size -= read_size
                blocks += 1
            text = buffer.decode(errors="ignore").splitlines()
            last_lines = "\n".join(text[-lines:])
            return PlainTextResponse(last_lines)
    except Exception as e:
        logger.exception(f"Error reading log file: {e}")
        return PlainTextResponse(f"Error reading log file: {e}", status_code=500)

@app.on_event("startup")
async def startup_event():
    async def keep_alive():
        while True:
            try:
                logger.info("[KEEPALIVE] Service heartbeat")
                flush_logs()
            except Exception:
                pass
            await asyncio.sleep(settings.KEEP_ALIVE_INTERVAL_SECONDS)
    asyncio.create_task(keep_alive())

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("[SHUTDOWN] Waiting for background tasks to finish (graceful shutdown)...")
    for t in background_tasks_list:
        if not t.done():
            try:
                t.cancel()
            except Exception:
                pass
    await asyncio.sleep(0.5)
    flush_logs()