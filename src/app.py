from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schema import ChatRequestSchema, FetchChatSchema
from uuid import uuid4
import datetime


# Dictionary to store chat history for each user
chats = {}

FEW_SHOT_PROMPT = """
You are a reasoning assistant. Answer questions to the point without being verbose. Examples:

Q: What is the capital of France?
A: Paris.

Q: Why does ice float on water?
A: Ice is less dense than water.

Q: What is 5 + 7?
A: 12.

Now, answer the following:
"""

FEW_SHOT_SUMMARY = """
You are a reasoning assistant. Summarize the conversation to the point without being verbose in one phrase. Examples:

Q: What is the capital of France?
A: Capital of France

Q: Why does ice float on water?
A: Ice Floating on Water

Q: What is 5 + 7?
A: Result of 5 + 7

Now, answer the following:
"""

# Initialize FastAPI application
app = FastAPI()

# Add CORS middleware to allow the frontend (localhost:3000) to interact with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load OpenAI API key from a file and set it for API requests
with open("../openai_api_key.txt", "r") as file:
    api_key = file.read().strip()

client = OpenAI(api_key=api_key)

################################################
# Endpoints
################################################


# Endpoint to start a new chat
@app.post("/new-chat")
def new_chat():
    chat_id = str(uuid4())
    return {"chatId": chat_id}


# Endpoint to process user input and retrieve chat history from ChatGPT API
@app.post("/chats")
async def request_chatgpt(request: ChatRequestSchema):
    dynamic_prompt = FEW_SHOT_PROMPT

    if request.chatId not in chats:
        chats[request.chatId] = {"messages": [], "topic": None}

    dynamic_prompt += f"Q: {request.prompt}"

    user_message = {
        "no": len(chats[request.chatId].get("messages", [])) + 1,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sender": "user",
        "content": request.prompt,
    }

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a supportive assistant."},
                *[
                    {"role": msg["sender"], "content": msg["content"]}
                    for msg in chats[request.chatId]["messages"]
                ],
                {"role": "user", "content": dynamic_prompt},
            ],
            max_tokens=100,
            temperature=0.7,
        )
        assistant_message = {
            "no": len(chats[request.chatId].get("messages", [])) + 2,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sender": "assistant",
            "content": response.choices[0].message.content.strip(),
        }

        chats[request.chatId]["messages"].append(user_message)
        chats[request.chatId]["messages"].append(assistant_message)

        dynamic_summary = FEW_SHOT_SUMMARY
        dynamic_summary += f"Q: {request.prompt}"

        if chats[request.chatId]["topic"] is None:
            try:
                topic_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a supportive assistant.",
                        },
                        {"role": "user", "content": dynamic_summary},
                    ],
                    max_tokens=50,
                    temperature=0.5,
                )
                chat_topic = topic_response.choices[0].message.content.strip()
                chats[request.chatId]["topic"] = chat_topic
            except Exception as e:
                print(f"Error summarizing chat topic: {e}")
                raise HTTPException(
                    status_code=500, detail="Error summarizing chat topic"
                )

        return {
            "messages": chats[request.chatId]["messages"],
            "topic": chats[request.chatId]["topic"],
        }
    except Exception as e:
        print(f"Error communication with OpenAI: {e}")
        raise HTTPException(status_code=500, detail="Error communicating with OpenAI")


@app.post("/fetch")
def fetchChat(request: FetchChatSchema):
    try:
        return {"messages": chats[request.chatId]["messages"]}
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"Chat ID {request.chatId} not found"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error communicating with OpenAI")
