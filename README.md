Some numbers on e5 (output dim 1024), ndcg@10 on NFCorpus. The auto-encoders are trained  on 160k cleaned passages of wikipedia.  

|  dim      | ndcg@10 | loss(1)  |
|-----------|---------|----------|
| e5 (orig) | 0.35661 |    -     |
| 1024(2)   | 0.33514 | 0.000104 |
| 768       | 0.33509 | 0.000106 |
| 512       | 0.33013 | 0.000108 |
| 256       | 0.32846 | 0.000117 |


(1) The loss here represents the peak reconstruction loss that led to the score.

(2) Training sanity check (is fine-tuning on wiki hurting the embeddings) 

## e5 -- mxbai:

* e5          : 0.35661
* mxbai       : 0.37716
* Naive concat: 0.37744

|  dim      | ndcg@10 | loss(1)  |
|-----------|---------|----------|
| 1024      | 0.37646 | 0.008700 |
| 768       | 0.37876 | 0.008813 |

For the 1024 compression, see how the val loss (on wikipedia) affects the ndcg@10:

| ndcg@10 | val loss |
|---------|----------|
| 0.36197 | 0.017030 |
| 0.36382 | 0.017025 |
| 0.3754  | 0.013774 |
| 0.37646 | 0.008700 |

(*) No auto-encoder, naive concat.

Focusing on the 768 compression as it would be needed later on to augment, a simple concat of bge-small and e5-small (0.35509). Things to try:

* Better loss functions (nope, did not work).
* Deeper auto-encoder (nope, did not work) >> shallow networks works best? weird, but ok!
* Batch of 64 slightly improved.


| ndcg@10 | val loss |
|---------|----------|
| 0.35667     | 0.018992 |
| 0.36019     | 0.017579 |
| 0.36951     | 0.016789 |
| 0.37037     | 0.013036 | 
| **0.37876** | 0.008813 | 

### Small models (<35M):

* Collect 2 models and check the mteb scores if ok.  [DONE]
    * m1: `intfloat/e5-small-v2`  (33M)
        * ndcg@10 - NFCorpus: 0.31806
    * m2: `BAAI/bge-small-en-v1.5`  (33M)
        * ndcg@10 - NFCorpus: 0.33708

* Naive concat (768):  0.35509

### Experiment: Can we improve naive concat. of small models with compressed (using auto-encoders) concat of large models?

1. Using e5-large and mxbai:
    * Concat e5-large and mxbai > `e5-mxbai-2048`.
    * Learn a 786-dim (auto-enc) representation  `e5-mxbai-autoenc-768`.

2. Using bge-smal and e5-small:
    * Concat e5-large and mxbai > `bge-e5-768`.

3. Map the output of `bge-e5-768` to `e5-mxbai-autoenc-768`
    * Using a simple FFNN.
    * Maybe use residual links


