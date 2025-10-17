from fastapi import FastAPI, HTTPException
from starlette.responses import JSONResponse
from models import TaskRequest
from config import get_settings
import asyncio
import httpx
import json
import os
import base64
import re
import git
import time
import shutil
import stat

settings = get_settings()
githubusername = settings.GITHUB_USERNAME
student_email = settings.STUDENT_EMAIL

GITHUB_API_BASE = "https://api.github.com"
GITHUB_PAGES_BASE = f"https://{githubusername}.github.io"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
GEMINI_API_KEY = settings.GEMINI_API_KEY

app = FastAPI(
    title="Automated Task Receiver & Processor",
    description="Endpoint for receiving task assignments and triggering AI code generation/deployment."
)

received_task_data = None

async def setup_local_repo(localpath: str, reponame: str, repourl_auth: str, repourl_http: str, roundindex: int):
    githubtoken = settings.GITHUB_TOKEN
    headers = {
        "Authorization": f"token {githubtoken}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    async with httpx.AsyncClient(timeout=45) as client:
        try:
            if roundindex == 1:
                payload = {"name": reponame, "private": False, "auto_init": True}
                response = await client.post(f"{GITHUB_API_BASE}/user/repos", json=payload, headers=headers)
                response.raise_for_status()
                repo = git.Repo.init(localpath)
                repo.create_remote("origin", repourl_auth)
            elif roundindex == 2:
                repo = git.Repo.clone_from(repourl_auth, localpath)
            return repo
        except httpx.HTTPStatusError as e:
            raise Exception("GitHub API call failed during repository setup.")
        except git.GitCommandError as e:
            raise Exception("Git operation failed during repository setup.")

async def commit_and_publish_repo(repo: git.Repo, taskid: str, roundindex: int, reponame: str):
    githubtoken = settings.GITHUB_TOKEN
    headers = {
        "Authorization": f"token {githubtoken}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    repourl_http = f"https://github.com/{githubusername}/{reponame}"
    async with httpx.AsyncClient(timeout=45) as client:
        try:
            repo.git.add(A=True)
            commitmessage = f"Task {taskid} - Round {roundindex} LLM-generated app update/creation"
            repo.index.commit(commitmessage)
            commitsha = repo.head.object.hexsha
            repo.git.branch("-M", "main")
            repo.git.push("--set-upstream", "origin", "main", force=True)
            await asyncio.sleep(10)
            pages_api_url = f"{GITHUB_API_BASE}/repos/{githubusername}/{reponame}/pages"
            pages_payload = {"source": {"branch": "main", "path": "/"}}
            pages_max_retries = 5
            pages_base_delay = 3
            for retry_attempt in range(pages_max_retries):
                try:
                    pages_response = await client.get(pages_api_url, headers=headers)
                    is_configured = pages_response.status_code == 200
                    if is_configured:
                        await client.put(pages_api_url, json=pages_payload, headers=headers)
                    else:
                        await client.post(pages_api_url, json=pages_payload, headers=headers)
                    break
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 422 and "main branch must exist" in e.response.text and retry_attempt < pages_max_retries - 1:
                        delay = pages_base_delay * 2 ** retry_attempt
                        await asyncio.sleep(delay)
                    else:
                        raise
            await asyncio.sleep(5)
            pages_url = f"{GITHUB_PAGES_BASE}/{reponame}/"
            return {"repourl": repourl_http, "commitsha": commitsha, "pagesurl": pages_url}
        except git.GitCommandError as e:
            raise Exception("Git operation failed during deployment.")
        except httpx.HTTPStatusError as e:
            raise Exception("GitHub API call failed during deployment.")
        except Exception as e:
            raise

def datauri_to_gemini_part(datauri: str):
    if not datauri or not datauri.startswith("data:"):
        return None
    try:
        match = re.search(r"data:(?P<mimetype>[\w/-]+);base64,(?P<base64data>.+)", datauri, re.IGNORECASE)
        if not match:
            return None
        mimetype = match.group("mimetype")
        base64data = match.group("base64data")
        if not mimetype.startswith("image"):
            return None
        return {"inlineData": {"data": base64data, "mimeType": mimetype}}
    except Exception as e:
        return None

def is_image_datauri(datauri: str):
    if not datauri.startswith("data:"):
        return False
    return re.search(r"data:image/.*;base64,", datauri, re.IGNORECASE) is not None

async def save_generated_files_locally(taskid: str, files: dict):
    basedir = os.path.join(os.getcwd(), "generatedtasks")
    taskdir = os.path.join(basedir, taskid)
    os.makedirs(taskdir, exist_ok=True)
    for filename, content in files.items():
        filepath = os.path.join(taskdir, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            raise Exception(f"Failed to save file {filename} locally.")
    return taskdir

async def call_llm_for_code(prompt: str, taskid: str, imageparts: list):
    normalizedprompt = prompt.lower().strip()
    if "captcha solver" in normalizedprompt and "responsive" not in normalizedprompt:
        prompt = (
            "Ensure the web app is a single, complete, fully responsive HTML file using Tailwind CSS. "
            "It must fetch an image from the query parameter ?url=https://.../image.png, display it, and perform OCR using Tesseract.js via CDN. "
            "If the URL parameter is missing, use the attached sample image by default. Show the recognized text and any errors clearly in the UI. "
            "Return output strictly as a JSON object with keys index.html, README.md, and LICENSE."
        )
    systemprompt = (
        "You are an expert full-stack engineer and technical writer. Your task is to generate three files in a single structured JSON response: "
        "index.html, README.md, and LICENSE. The index.html must be a single, complete, fully responsive HTML file using Tailwind CSS for styling and must implement the requested application logic. "
        "The README.md must be professional. The LICENSE must contain the full text of the MIT license."
    )
    responseschema = {
        "type": "OBJECT",
        "properties": {
            "index.html": {"type": "STRING", "description": "The complete, single-file HTML content with inline CSS and JS, using Tailwind."},
            "README.md": {"type": "STRING", "description": "The professional Markdown content for the project README."},
            "LICENSE": {"type": "STRING", "description": "The full text of the MIT license."}
        },
        "required": ["index.html", "README.md", "LICENSE"]
    }
    contents = []
    if imageparts:
        allparts = imageparts + [{"text": prompt}]
        contents.extend(allparts)
    else:
        contents.append({"text": prompt})
    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": systemprompt}]},
        "generationConfig": {},
        "responseMimeType": "application/json",
        "responseSchema": responseschema
    }
    maxretries = 3
    basedelay = 1
    for attempt in range(maxretries):
        try:
            url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
                response.raise_for_status()
                result = response.json()
                jsontext = result["candidates"][0]["content"]["parts"][0]["text"]
                generatedfiles = json.loads(jsontext)
                return generatedfiles
        except (httpx.HTTPStatusError, httpx.RequestError, KeyError, json.JSONDecodeError) as e:
            if attempt < maxretries - 1:
                delay = basedelay * 2 ** attempt
                await asyncio.sleep(delay)
            else:
                raise Exception("LLM Code Generation Failure")

@app.post("/ready", status_code=200)
async def receive_task(taskdata: TaskRequest):
    global received_task_data
    if taskdata.secret != settings.STUDENT_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized: Secret does not match configured student secret.")
    received_task_data = taskdata.dict()
    asyncio.create_task(generate_files_and_deploy(taskdata))
    return JSONResponse(status_code=200, content={"status": "ready", "message": f"Task {taskdata.task} received and processing started."})

@app.get("/")
async def root():
    return {"message": "Task Receiver Service is running. Post to /ready to submit a task."}

@app.get("/status")
async def get_status():
    global received_task_data
    if received_task_data:
        return received_task_data
    return {"status": "idle"}

async def generate_files_and_deploy(taskdata: TaskRequest):
    taskid = taskdata.task
    email = taskdata.email
    roundindex = taskdata.round
    brief = taskdata.brief
    evaluationurl = taskdata.evaluationurl
    nonce = taskdata.nonce
    attachments = taskdata.attachments
    localpath = os.path.join(os.getcwd(), "generatedtasks", taskid)
    if os.path.exists(localpath):
        def onerror(func, path, excinfo):
            if excinfo[0] is PermissionError or "WinError 5" in str(excinfo[1]):
                os.chmod(path, stat.S_IWUSR)
                func(path)
            else:
                raise
        try:
            shutil.rmtree(localpath, onerror=onerror)
        except Exception as e:
            raise Exception(f"Failed to perform local cleanup: {e}")
    imageparts = []
    attachmentlistforllmprompt = []
    for attachment in attachments:
        filename = attachment.name
        datauri = attachment.url
        if not filename or not datauri or not datauri.startswith("data:"):
            continue
        if is_image_datauri(datauri):
            part = datauri_to_gemini_part(datauri)
            if part:
                imageparts.append(part)
        attachmentlistforllmprompt.append(filename)
    llmprompt = f"Generate a complete, single-file HTML web application to achieve the following brief: {brief}"
    if attachmentlistforllmprompt:
        llmprompt += f"\n\nContext: The following files are available in the project root: {', '.join(attachmentlistforllmprompt)}. Ensure your code references these files correctly if applicable."
    generatedfiles = await call_llm_for_code(llmprompt, taskid, imageparts)
    await save_generated_files_locally(taskid, generatedfiles)
    reponame = taskid.replace(" ", "-").lower()
    githubtoken = settings.GITHUB_TOKEN
    repourl_auth = f"https://{githubusername}:{githubtoken}@github.com/{githubusername}/{reponame}.git"
    repourl_http = f"https://github.com/{githubusername}/{reponame}.git"
    try:
        repo = await setup_local_repo(localpath, reponame, repourl_auth, repourl_http, roundindex)
        for filename, content in generatedfiles.items():
            filepath = os.path.join(localpath, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
        deploymentinfo = await commit_and_publish_repo(repo, taskid, roundindex, reponame)
        repourl = deploymentinfo["repourl"]
        commitsha = deploymentinfo["commitsha"]
        pagesurl = deploymentinfo["pagesurl"]
        await notify_evaluation_server(evaluationurl, email, taskid, roundindex, nonce, repourl, commitsha, pagesurl)
    except Exception as e:
        pass

async def notify_evaluation_server(evaluationurl: str, email: str, taskid: str, roundindex: int, nonce: str, repourl: str, commitsha: str, pagesurl: str):
    payload = {
        "email": student_email,
        "task": taskid,
        "round": roundindex,
        "nonce": nonce,
        "repo_url": repourl,
        "commit_sha": commitsha,
        "pages_url": pagesurl
    }
    maxretries = 3
    basedelay = 1
    for attempt in range(maxretries):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(evaluationurl, json=payload)
                response.raise_for_status()
                return True
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            if attempt < maxretries - 1:
                delay = basedelay * 2 ** attempt
                await asyncio.sleep(delay)
            else:
                return False