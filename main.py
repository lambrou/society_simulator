from transformers import LlamaForCausalLM, LlamaTokenizer

if __name__ == '__main__':

    tokenizer = LlamaTokenizer.from_pretrained("/output/path")
    model = LlamaForCausalLM.from_pretrained("/output/path")