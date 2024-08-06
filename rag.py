import os
import asyncio
import aiohttp
from datetime import datetime
import time
import json
from langchain_community.vectorstores import Chroma
from dateutil import parser
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Script started. Initializing RAG pipeline and Farcaster bot...")

# Constants

class RAGPipeline:
    def __init__(self):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.model = ChatOllama(model="llama3")
        
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            You are an AI assistant specializing in summarizing and answering questions about the podcast conversation between Lex Fridman and Elon Musk titled "Neuralink and the Future of Humanity". Your knowledge comes from a transcript of this 8-hour conversation.

            Here are your instructions:
            1. Use the provided context to answer the user's question accurately and concisely.
            2. Focus on information directly from the conversation. Do not introduce external information or speculation.
            3. If the question is about a topic not covered in the conversation, state that it wasn't discussed in this podcast.
            4. If asked for opinions, provide Elon Musk's or Lex Fridman's views as expressed in the conversation, not your own.
            5. Summarize complex topics in simple terms, as if explaining to a general audience.
            6. If a question is ambiguous, interpret it in the context of the podcast topics: Neuralink, AI, space exploration, and the future of humanity.
            7. Your response should be between 100 and 1000 characters for Farcaster compatibility.
            8. Begin your response with a brief phrase that connects to the question, then provide the answer.

            Context from the transcript:
            {context}

            Human: {question}

            AI Assistant: Let's address that based on the Lex Fridman and Elon Musk conversation.
            """
        )

    def ingest(self, file_path: str):
        with open(file_path, 'r') as file:
            text = file.read()
        chunks = self.text_splitter.split_text(text)

        vector_store = Chroma.from_texts(texts=chunks, embedding=FastEmbedEmbeddings())
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def ask(self, query: str):
        if not hasattr(self, 'chain'):
            return "Please ingest the document first."
        
        response = self.chain.invoke(query)
        
        if "wasn't discussed in this podcast" in response:
            return f"I apologize, but that topic wasn't discussed in the Lex Fridman and Elon Musk podcast. Is there another aspect of their conversation about Neuralink, AI, or the future of humanity you'd like to know about?"
        
        return response

class FarcasterBot:
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
        self.last_checked_timestamp = datetime.now().timestamp() * 1000
        self.processed_replies = set()
        self.NEYNAR_API_KEY = ""
        self.NEYNAR_SIGNER = ""
        self.ORIGINAL_CAST_URL = ""  # Replace with your cast URL

    async def check_new_replies(self):
        logger.info(f"Checking for new replies for cast: {self.ORIGINAL_CAST_URL}")
        cursor = None
        
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    params = {
                        "identifier": self.ORIGINAL_CAST_URL,
                        "type": "url",
                        "reply_depth": 2,
                        "include_chronological_parent_casts": "false",  # Changed to string "false"
                        "limit": 50
                    }
                    if cursor:
                        params["cursor"] = cursor
                    
                    logger.info(f"Sending API request with params: {params}")
                    async with session.get(
                        "https://api.neynar.com/v2/farcaster/cast/conversation",
                        params=params,
                        headers={
                            "accept": "application/json",
                            "api_key": self.NEYNAR_API_KEY
                        }
                    ) as response:
                        if response.status != 200:
                            logger.error(f"API request failed with status {response.status}: {await response.text()}")
                            break
                        data = await response.json()
                    
                    logger.info(f"API response received: {json.dumps(data, indent=2)}")

                if "conversation" not in data or "cast" not in data["conversation"]:
                    logger.error(f"Unexpected API response structure: {json.dumps(data, indent=2)}")
                    break

                conversation = data["conversation"]["cast"]
                new_replies = [reply for reply in conversation.get("direct_replies", []) 
                            if parser.parse(reply["timestamp"]).timestamp() * 1000 > self.last_checked_timestamp
                            and reply["hash"] not in self.processed_replies]

                logger.info(f"Found {len(new_replies)} new replies")

                for reply in new_replies:
                    logger.info(f"Processing reply: {reply['hash']}")
                    await self.process_cast(reply)
                    self.processed_replies.add(reply["hash"])

                if new_replies:
                    self.last_checked_timestamp = max(parser.parse(reply["timestamp"]).timestamp() * 1000 for reply in new_replies)
                    logger.info(f"Updated lastCheckedTimestamp to: {datetime.fromtimestamp(self.last_checked_timestamp / 1000).isoformat()}")

                if "next" in data and data["next"].get("cursor"):
                    cursor = data["next"]["cursor"]
                    logger.info(f"Moving to next page with cursor: {cursor}")
                else:
                    logger.info("No more pages to process")
                    break
            except Exception as e:
                logger.exception(f"Error in check_new_replies: {str(e)}")
                break

        async def process_cast(self, cast):
            print(f"[{datetime.now().isoformat()}] Processing cast with hash: {cast['hash']}")
            
            if cast["text"]:
                print(f"[{datetime.now().isoformat()}] Processing cast text: {cast['text']}")
                response = self.rag.ask(cast["text"])
                
                while len(response) > 1001 or len(response) < 100:
                    print(f"[{datetime.now().isoformat()}] Response length inappropriate, trying again...")
                    response = self.rag.ask(cast["text"])

                print(f"[{datetime.now().isoformat()}] Final response: {response}")

                cast_options = {
                    "text": response,
                    "embeds": [],
                    "parent": cast["hash"],
                    "signer_uuid": self.NEYNAR_SIGNER,
                }
                print(f"[{datetime.now().isoformat()}] Preparing to publish cast with options: {cast_options}")

                await self.publish_cast_to_the_protocol(cast_options)
            else:
                print(f"[{datetime.now().isoformat()}] Cast does not contain text")

            print(f"[{datetime.now().isoformat()}] Finished processing cast {cast['hash']}")

        async def publish_cast_to_the_protocol(self, cast_options):
            async with aiohttp.ClientSession() as session:
                while True:
                    try:
                        async with session.post(
                            "https://api.neynar.com/v2/farcaster/cast",
                            json=cast_options,
                            headers={"api_key": self.NEYNAR_API_KEY}
                        ) as response:
                            data = await response.json()
                            print(f"[{datetime.now().isoformat()}] Published cast response: {data}")
                            return data["cast"]
                    except Exception as e:
                        print(f"[{datetime.now().isoformat()}] Error publishing cast, retrying in 60 seconds: {str(e)}")
                        await asyncio.sleep(60)

async def main():
    try:
        logger.info("Initializing RAG pipeline...")
        rag = RAGPipeline()
        logger.info("Ingesting audio.txt...")
        rag.ingest("audio.txt")
        logger.info("Initializing FarcasterBot...")
        bot = FarcasterBot(rag)

        logger.info("Starting main loop to check for new replies...")
        while True:
            try:
                await bot.check_new_replies()
            except Exception as e:
                logger.exception(f"Error in check_new_replies: {str(e)}")
                logger.info("Waiting 161 seconds before retrying...")
                await asyncio.sleep(161)  # Wait 161 seconds before retrying
            else:
                logger.info("Waiting 30 seconds before next check...")
                await asyncio.sleep(30)  # Check every 30 seconds if successful
    except Exception as e:
        logger.exception(f"Fatal error in main loop: {str(e)}")
        raise

if __name__ == "__main__":
    while True:
        try:
            asyncio.run(main())
        except Exception as e:
            logger.error(f"Main loop crashed. Restarting in 60 seconds. Error: {str(e)}")
            time.sleep(60)