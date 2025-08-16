from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50,  # Generate up to 50 new tokens
    truncation=True,
    pad_token_id=tokenizer.eos_token_id
)