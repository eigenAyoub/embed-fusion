from sentence_transformers import SentenceTransformer
import mteb 
import sys

models = {
            "mxbai"     : "mixedbread-ai/mxbai-embed-large-v1",  
            "bge"       : "BAAI/bge-large-en-v1.5"                 ,
            "e5"        : "intfloat/e5-large-v2"              ,
            "snowflake-m" : "Snowflake/snowflake-arctic-embed-m",
            "snowflake-l" : "Snowflake/snowflake-arctic-embed-l",
            "gte-base"        : "thenlper/gte-base",
            "gte-large"       : "thenlper/gte-large",
            "gte-small"       : "thenlper/gte-small",
            "e5-small"        : "intfloat/e5-small-v2", # (33M)
            "bge-small"       : "BAAI/bge-small-en-v1.5" # (33M)
}




m_name = sys.argv[1]

print(f"Eval for {m_name}")

model = SentenceTransformer(models[m_name])
tasks = mteb.get_tasks(tasks=["NFCorpus"]) 
evaluation = mteb.MTEB(tasks=tasks, eval_splits=["test"], metric="ndcg@10")
results_m1 = evaluation.run(model, output_folder = f"results/{m_name}_ST")