from pydantic import BaseModel, Field, EmailStr
from typing import List

class Attachment(BaseModel):
    name: str = Field(..., description="Name of the attached file (e.g., 'sample.png')")
    url: str = Field(..., description="The content encoded as a data URI (data:image/png;base64,...)")

class TaskRequest(BaseModel):
    email: EmailStr = Field(..., description="Student email ID")
    secret: str = Field(..., description="Student-provided secret")
    task: str = Field(..., description="A unique task ID (e.g., 'captcha-solver-...')")
    round: int = Field(..., description="The round index (e.g., 1)")
    nonce: str = Field(..., description="Pass this nonce back to the evaluation URL below")
    brief: str = Field(..., description="Brief description of what the app needs to do")
    checks: List[str] = Field(..., description="Evaluation checks (e.g., license, readme quality)")
    evaluation_url: str = Field(..., description="URL to send repo & commit details")
    attachments: List[Attachment] = Field(..., description="Attachments encoded as data URIs")
