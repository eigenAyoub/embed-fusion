import click
import os
import mteb
import pandas as pd
from mteb import MTEB
from ef.config import *
from ef.globals import *
from ef.utils import *


class CombinedRetrievalModel():
    def __init__(self, models: List[RetrievalModel]) -> None:
        self.models = models

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        list_embeddings = []
        for model in self.models:
            list_embeddings.append(model.encode_queries(queries, **kwargs))
        return np.concatenate(list_embeddings, axis=1)
    
    def encode_corpus(self, corpus: List[str], **kwargs) -> np.ndarray:
        list_embeddings = []
        for model in self.models:
            list_embeddings.append(model.encode_corpus(corpus, **kwargs))
        return np.concatenate(list_embeddings, axis=1)


@click.command()
def evaluate_two():
    """
    Evaluate two.
    """
    click.echo('Evaluate all combinations of two models.')

    for task_name in TASK_NAMES:
        for i, p1 in enumerate(EMBEDDING_MODELS):
            for j, p2 in enumerate(EMBEDDING_MODELS):
                if i == j:
                    _evaluate(task_name, [p1])
                if i < j:
                    _evaluate(task_name, [p1, p2])


@click.command()
@click.argument("task_name", type=click.Choice(TASK_NAMES), default=TASK_NAMES[0])
@click.argument("pretrained_model_names", nargs=-1, type=click.Choice(EMBEDDING_MODELS + ['all']))
def evaluate(task_name: str, pretrained_model_names: List[str]):
    """
    Evaluate.
    """
    _evaluate(task_name, pretrained_model_names)


def _evaluate(task_name: str, pretrained_model_names: List[str]):
    if len(pretrained_model_names) == 0:
        pretrained_model_names = (EMBEDDING_MODELS[0],)
    elif 'all' in pretrained_model_names:
        pretrained_model_names = EMBEDDING_MODELS
    
    pretrained_model_names = sorted(pretrained_model_names)

    click.echo(f'Task name: {task_name}')
    click.echo(f'Pretrained model names: {pretrained_model_names}')

    eval_split = 'dev' if 'msmarco' in task_name.lower() else 'test'

    models = [RetrievalModel(p, task=task_name) for p in pretrained_model_names]
    combined_model = CombinedRetrievalModel(
        models=models
    )

    combined_model_name = '___'.join([p.replace('/', '_') for p in pretrained_model_names])
    task_output_path = os.path.join('results', combined_model_name, 'no_autoencoder')
    results_file_path = os.path.join(task_output_path, 'no_model_name_available', 'no_revision_available', f'{task_name}.json')

    task = mteb.get_task(task_name)
    evaluation = MTEB(tasks=[task])
    evaluation.run(combined_model,
            eval_splits=[eval_split],
            output_folder=task_output_path,
            override_results=True,
            encode_kwargs={'batch_size': 32},
    )

    # Create the evaluation results
    index = pd.MultiIndex.from_tuples(
            [(task_name, combined_model_name)],
            names=['task_name', 'combined_model_name'])
    df_results = pd.DataFrame(index=index, columns=['NDCG@10'])

    # Read json file
    import json
    with open(results_file_path, 'r') as file:
        data = json.load(file)
        ndcg_at_10 = data['scores'][eval_split][0]['ndcg_at_10']
        click.echo(f'NDCG@10 for task {task_name}: {ndcg_at_10}')
        df_results.at[(task_name, combined_model_name), 'NDCG@10'] = ndcg_at_10

    # Save results
    update_meta(df_results, RESULTS_CONCAT_FILE)