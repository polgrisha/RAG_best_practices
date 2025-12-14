from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch


def load_llama(model_name, quantize=True):    
    # quantization_config reduces memory usage to fit on consumer GPUs (Colab T4, etc.)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if quantize:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto"
            )
    except OSError as e:
        print("\nERROR: Could not load model. Did you set your HF_TOKEN and accept the license on Hugging Face?")
        raise e
        
    # Ensure pad token is set (Llama 3 often lacks a default pad token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    # Explicitly update the model's generation config to suppress warnings
    model.generation_config.pad_token_id = tokenizer.pad_token_id
        
    return tokenizer, model


def create_prompts_nq(tokenizer, questions, contexts=None):
    """
    Creates prompts for NQ Open. 
    If contexts are provided (RAG), constrains the model to use them.
    If contexts are None (Closed Book), relies on internal knowledge.
    """
    prompts = []
    
    # If no contexts provided, treat as list of None for zipping
    if contexts is None:
        contexts = [None] * len(questions)
        
    for q, ctx in zip(questions, contexts):
        if ctx:
            # RAG Prompt
            system_content = "Answer the question using only the provided context. Answer with a short phrase or entity name only."
            user_content = f"Context:\n{ctx}\n\nQuestion: {q}"
        else:
            # Closed Book Prompt
            system_content = "Answer the question with a short phrase or entity name only. Do not use complete sentences."
            user_content = f"Question: {q}"
            
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    return prompts


def generate_batch(model, tokenizer, prompts):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    # Llama 3 specific terminators to ensure it stops generating immediately
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20, # Short answers (NQ style)
            eos_token_id=terminators,
            do_sample=False,   # Greedy decoding for evaluation
            temperature=None,
            top_p=None
        )
    
    # Slice off the prompt to get just the answer
    generated_ids = outputs[:, inputs.input_ids.shape[1]:]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)