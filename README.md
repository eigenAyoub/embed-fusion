Fill this.



### Run:

* For distillation, try a deeper network?
    * Drop the student > Deep network.



* Back to a baseline with teacher:

* Step 1: Only between, output layer.
    * SimilarityLoss: 
    * 768 -> 1024 -> 768

* Step 2: [hidden layer -- teacher] + [output-layer -- student]
    * SimilarityLoss: Only between, output layer.
    * 768 -> 1024 -> 768 (early test, very quick to saturate. Stuck at .33 and .70)
    * 1 MRL dim > 768

    * with residual / and without




## Observations:

* On trying other losses:
    * **Nothing** works better than MSE on matrix-similarity, e.g.: 
    * Contrastive Loss.
    * KL loss and its variants.

* Include teacher  (mxbai) with KL loss.
    * Not working with `e5-small + no-instruct` as a student model.
    * Not improving scores on `no-ins + gte-small`

* Baseline: Drop the teacher, focus on encoding better `e5-small + no-instruct`:
    * Higher batch size (64 -> 128 -> 256) -- Higher seems to be better. 
    * lr, yes do change that 
    * high / loss / mid MRL.

* `no-ins` (384) itself is fine-tune of `bge-small` (512); drop, include the original `bge-small` and see.

* Work with a 7B teacher (`linq`).






* all-33M mse on similarity matrix.
* all-33M mse on softmax + similarity matrix.

* try `pin_memory = True` see if it helps with memory loading -- Nope.