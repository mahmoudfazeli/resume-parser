from fastapi import FastAPI, File, UploadFile, HTTPException
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.together import TogetherLLM
from llama_index.embeddings.together import TogetherEmbedding
from llama_parse import LlamaParse
from dotenv import load_dotenv
import nest_asyncio
import os

app = FastAPI()

nest_asyncio.apply()
load_dotenv()

llm = TogetherLLM(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1", 
    api_key=os.environ["TOGETHER_API_KEY"]
)
embed_model = TogetherEmbedding(
    model_name="togethercomputer/m2-bert-80M-8k-retrieval", 
    api_key=os.environ["TOGETHER_API_KEY"]
)

Settings.llm = llm
Settings.embed_model = embed_model

parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    result_type="markdown",
    verbose=True,
)

@app.post("/analyze-cv/")
async def analyze_cv(file: UploadFile = File(...)):
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=415, detail="Unsupported file type.")
    input_cv = await parser.load_data(await file.read())  # Assuming this method can handle file streams

    # You may need to handle how the PDF is converted to text here
    index = VectorStoreIndex.from_documents(input_cv)

    query_engine = index.as_query_engine()
    response = query_engine.query(
        "You are a brilliant career adviser. Answer a question of job seekers with given information.\n"
        "If their CV information is given, use that information as well to answer the question.\n"
        "If you are asked to return jobs that are suitable for the job seeker, return Job ID, Title and Link.\n"
        "If you are not sure about the answer, return NA. \n"
        "You need to show the source nodes that you are using to answer the question at the end of your response.\n"
        f"CV: {input_cv[0]} \n"
    )
    return {"response": response.response}

