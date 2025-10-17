from pydantic import BaseModel, Field, EmailStr
from typing import List

# Data model for attachments (like the sample image)
class Attachment(BaseModel):
    name: str = Field(..., description="Name of the attached file (e.g., 'sample.png')")
    url: str = Field(..., description="The content encoded as a data URI (data:image/png;base64,...)")

# The main data model for the incoming task payload
class TaskRequest(BaseModel):
    # Student email ID
    email: EmailStr = Field(..., description="Student email ID")
    # Student-provided secret
    secret: str = Field(..., description="Student-provided secret")
    # A unique task ID.
    task: str = Field(..., description="A unique task ID (e.g., 'captcha-solver-...')")
    # There will be multiple rounds per task. This is the round index
    round: int = Field(..., description="The round index (e.g., 1)")
    # Pass this nonce back to the evaluation URL
    nonce: str = Field(..., description="Pass this nonce back to the evaluation URL below")
    # brief: mentions what the app needs to do
    brief: str = Field(..., description="Brief description of what the app needs to do")
    # checks: mention how it will be evaluated
    checks: List[str] = Field(..., description="Evaluation checks (e.g., license, readme quality)")
    # Send repo & commit details to the URL below
    evaluation_url: str = Field(..., description="URL to send repo & commit details")
    # Attachments will be encoded as data URIs
    attachments: List[Attachment] = Field(..., description="Attachments encoded as data URIs")



# from pydantic import BaseModel, EmailStr
# from typing import List, Optional

# # Defines the structure for an individual attachment, like a sample captcha image
# class Attachment(BaseModel):
#     """
#     Represents an attachment provided in the task payload.
#     The 'url' is expected to be a data URI (e.g., base64 encoded image).
#     """
#     name: str
#     url: str

# # Defines the complete structure of the JSON request body
# class TaskRequest(BaseModel):
#     """
#     The main model representing the task request sent by the evaluation server.
#     """
#     email: EmailStr  # Enforces a valid email format
#     secret: str
#     task: str
#     round: int
#     nonce: str
#     brief: str
#     checks: List[str]  # A list of strings detailing the evaluation checks
#     evaluation_url: str
#     attachments: List[Attachment] # A list of Attachment objects

#     # Configuration for Pydantic to allow validation from dicts/JSON
#     class Config:
#         schema_extra = {
#             "example": {
#                 "email": "student@example.com",
#                 "secret": "my-secure-token",
#                 "task": "captcha-solver-12345",
#                 "round": 1,
#                 "nonce": "ab12-cd34-ef56",
#                 "brief": "Create a captcha solver that handles ?url=https://.../image.png.",
#                 "checks": [
#                     "Repo has MIT license",
#                     "README.md is professional"
#                 ],
#                 "evaluation_url": "https://example.com/notify",
#                 "attachments": [
#                     {
#                         "name": "sample.png",
#                         "url": "data:image/png;base64,iVBORw..."
#                     }
#                 ]
#             }
#         }
