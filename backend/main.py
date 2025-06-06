import logging
import os
from  fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from semantic_kernel.functions import KernelArguments
from pydantic import BaseModel
import sys
import asyncio
from dotenv import load_dotenv


ENV_FILE = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(ENV_FILE)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow all origins for CORS
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers               
)

class chatRequest(BaseModel):
    prompt: str


@app.post("/chat")
async def chat_with_agent(request: chatRequest)-> dict:
    """
    Endpoint to chat with the agent using the provided prompt.
    """
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    AVAILABLE_MODELS: dict[str, AzureChatCompletion] = {
"default": AzureChatCompletion(
    service_id="default",
    ad_token_provider=token_provider,
    deployment_name=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],  # Ensure this environment variable is set",
    endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],  # Ensure this environment variable is set
    # endpoint="https://openai-semantic-kernel-dev.openai.azure.com/",
    api_version="2024-12-01-preview"  # Specify the API version if needed
),
    }
    kernel = Kernel()
    for service_id, service in AVAILABLE_MODELS.items():
        print(f"Adding service: {service_id} with model {service}")
        kernel.add_service(service)
    plugin = kernel.add_plugin(parent_directory="./prompt_templates/", plugin_name="FunPlugin")

    joke_function = plugin["Joke"]


    joke = await kernel.invoke(
        joke_function,
        KernelArguments(prompt=request, style="super silly"),
    )

    response = joke.model_dump()
    # print(f"Joke response: {response}")
    return {"response": response['value'][0]['items'][0]['text']}

    # except Exception as e:
    #     logging.error(f"Error in chat_with_agent: {e}")
    #     return {"error": str(e)}





if __name__ == "__main__":
    asyncio.run(app, reload=True, port=8000)  # Run the FastAPI app with auto-reload