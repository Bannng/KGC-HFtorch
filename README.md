# Knowledge Grounded Dialogue System Implementation with HuggingFace and PyTorch

### My implementation 
- package list
* conda install pytorch cudatoolkit=11.3 -c pytorch
* pip install transformers
* pip install parlai
* pip install colorlog
* pip install git+https://github.com/bckim92/language-evaluation.git


### Batch size tip
- per_device_train_batch_size : 1
- per_device_eval_batch_size : 1
- grad_accumulation_steps : 4
* which is available for 2080ti (nearly 110000MB)