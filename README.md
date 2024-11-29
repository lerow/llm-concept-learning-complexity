# llm-concept-learning-complexity
This repo contains the code needed to reproduce main experiment results in **Boolean complexity of concept learning**. All models are from [Hugging Face](https://huggingface.co/).

## Usage
To run the main experiments (gemma-2-9b-it, gemma-2-27b-it, Qwen2-7B-Instruct, Qwen2-72B-Instruct) using data in `data/`, and print results to txt files:
```
bash run-experiment.sh
```

Note that the code has not been tested on non-Nvidia GPUs. Qwen2-72B-Instruct experiments use 8-bit quantization from [bitsandbytes](https://pypi.org/project/bitsandbytes/).
For best performance, use GPUs with combined VRAM > 80GB.


## Dependencies
Required dependencies are listed in `requirements.txt`.



## Generate concept data
The data used in the experiments is in `data/`. To generate your own concept data, check `src/generate_concepts.py`. (documentation to be added)
