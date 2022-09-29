# Knowledge Grounded Dialogue System Implementation with HuggingFace and PyTorch

### My implementation 
- package list
* conda create -n kgc-torch python==3.8
* conda install pytorch cudatoolkit=11.3 -c pytorch
* pip install transformers (4.22.2)
* pip install parlai (1.7.1)
* pip install colorlog (6.7.0)
* pip install git+https://github.com/bckim92/language-evaluation.git


### Batch size tip
- per_device_train_batch_size : 1
- per_device_eval_batch_size : 1
- grad_accumulation_steps : 4
* which is available for 2080ti (nearly 110000MB)