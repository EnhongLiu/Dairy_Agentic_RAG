import mlflow

with mlflow.start_run() as run_id:
    model_info = mlflow.langchain.log_model(
        lc_model="Agentic_RAG_JDS_Web.py", # Path to model Python file 
        artifact_path="Agentic_RAG_JDS_web_langgraph",
        loader_fn = load_retriever,
        registered_model_name="RAG_JDS_web",
        pip_requirements=["langchain_community", 'faiss-cpu','langchain','langchain_databricks','databricks-langchain','google-ai-generativelanguage', 'langchain-google-genai','langgraph']
    )

    model_uri = model_info.model_uri

print(model_uri)
