## TODO:

* Check correctness for all models
    * e5 / no-instruct / mxbai



### Run:

* Include teacher  (mxbai) with KL loss.
    * Not working with `e5-small + no-instruct` as a student model.

* Baseline: Drop the teacher, focus on encoding better `e5-small + no-instruct`:
    * Higher batch size (64 -> 128 -> 256)
    * lr, yes do change that 
    * contrastive loss function 
    * regularize better.  
    * high / loss / mid MRL.

* `no-ins` (384) itself is fine-tune of `bge-small` (512); drop, include the original `bge-small` and see.

* Work with a 7B teacher (`linq`).

* Topics:



