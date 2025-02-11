import os
from langchain_google_genai import ChatGoogleGenerativeAI
from databricks_langchain import DatabricksEmbeddings
from transformers import AutoModelForCausalLM
from databricks_langchain import ChatDatabricks
import torch


os.environ['GOOGLE_API_KEY'] = 'xxxx' ################## Hide ##########################
os.environ['HF_TOKEN'] = 'xxxx' ######################### Hide ##########################
os.environ["DATABRICKS_TOKEN"]= 'xxxxx' ################## Hide ##########################

# Retriever Model
llama70b = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct",
                        temperature=0.0,
                        max_tokens=512)

# Judge Model (Use gemini2.0 for now)
gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp",
                             temperature=0.0,
                             max_tokens=512)
                                     
          
mixtral = ChatDatabricks(endpoint="databricks-mixtral-8x7b-instruct",
                        temperature=0.0,
                        max_tokens=512)

# Decision Model (use llama70b for now)
llama70b = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct",
                        temperature=0.0,
                        max_tokens=512)

# Embedding model
db_embeddings = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
