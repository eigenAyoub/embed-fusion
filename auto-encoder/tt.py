from sentence_transformers import SentenceTransformer
import mteb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DebugSentenceTransformer(SentenceTransformer):
    def encode(self, sentences, **kwargs):
        # Debug print to see incoming kwargs
        logger.info(f"Original kwargs: {kwargs}")

        print("init> ", kwargs)
        
        kwargs['prompt_name'] = 'query'
        logger.info("Setting prompt_name='query' for arctic model")
        
        logger.info(f"Final kwargs being used: {kwargs}")
        print("final > ", kwargs)
        return super().encode(sentences, **kwargs)

model_id = "Snowflake/snowflake-arctic-embed-m-v1.5"

model = DebugSentenceTransformer(model_id, 
                               trust_remote_code=True
                               ).to("cuda")

tasks = mteb.get_tasks(tasks=["NFCorpus"]) 

evaluation = mteb.MTEB(tasks=tasks, 
                      eval_splits=["test"], 
                      metric="ndcg@10")

results = evaluation.run(model, 
                        output_folder="res/snow_score33",
                        batch_size=128)

logger.info(f"Evaluation results: {results}")