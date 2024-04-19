from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import re
import uvicorn
import nest_asyncio
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.together import TogetherLLM
from llama_index.embeddings.together import TogetherEmbedding
from dotenv import load_dotenv
import tempfile

app = FastAPI()

# CORS configuration to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),  # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="markdown",  # "markdown" and "text" are available
    verbose=True,
)

def create_slug(input_string):
    string_low = input_string.lower()
    string_clean = re.sub(r'[^a-z0-9\s-]', '', string_low)
    slug = re.sub(r'\s+', '-', string_clean).strip('-')
    return slug

@app.post("/analyze-cv/")
async def analyze_cv(file: UploadFile = File(...)):
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=415, detail="Unsupported file type.")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()  # Ensure all data is written to disk

        input_cv = parser.load_data(file_path)

    index = VectorStoreIndex.from_documents(input_cv)
    query_engine = index.as_query_engine()
    response = query_engine.query(
        "You are a brilliant career adviser. Answer a question of job seekers with given information.\n"
        "If their CV information is given, use that information as well to answer the question.\n"
        "If you are asked to return jobs that are suitable for the job seeker, return Job ID, Title and Link.\n"
        "If you are not sure about the answer, return NA. \n"
        "You need to show the source nodes that you are using to answer the question at the end of your response.\n"
        f"CV: {input_cv[0]} \n"  # Adjust based on how `input_cv_text` is structured
    )

    slug = create_slug(response.response)
    api_response = {
        "responseText": response.response,
        "metadata": response.metadata,
        "sources": response.get_formatted_sources() if hasattr(response, 'get_formatted_sources') else None,
        "slug": slug
    }
    return JSONResponse(content=api_response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
