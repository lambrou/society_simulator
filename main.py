from transformers import LlamaForCausalLM, LlamaTokenizer

if __name__ == '__main__':
    tokenizer = LlamaTokenizer.from_pretrained("llama/tokenizer.model")
    model = LlamaForCausalLM.from_pretrained("llama/Llama-2-7b-chat-hf", load_in_8bit=True, device_map=0)
    while True:
        user_input = input("Enter your input: ")
        if user_input == "exit":
            break
        input_ids = tokenizer.encode(user_input, return_tensors="pt")
        output = model.generate(input_ids, max_length=1000, do_sample=True, top_p=0.95, top_k=60)
        print(tokenizer.decode(output[0], skip_special_tokens=True))
