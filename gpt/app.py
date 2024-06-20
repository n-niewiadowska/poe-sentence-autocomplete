from transformers import GPT2LMHeadModel, GPT2Tokenizer


def clean_text(text): 
  if text.startswith('Ġ'):
    text = text[1:]
  
  text = text.replace('Ċ', ' ').replace('\n', ' ')
  text = text.replace('Ġ', ' ').replace('�', '')
  
  return text.strip()

def run_app_gpt(model_dir):
  model = GPT2LMHeadModel.from_pretrained(model_dir, local_files_only=True)
  tokenizer = GPT2Tokenizer.from_pretrained(model_dir, local_files_only=True)

  print("Enter 'exit' to close the GPT-2 app.")
  
  while True:
    prompt = input("Enter your prompt:  ")

    if prompt.lower() == "exit":
      print("Closing the app...")
      break
  
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=200, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = clean_text(response)
    print("Generated text:")
    print(f"\" {response} \"")

