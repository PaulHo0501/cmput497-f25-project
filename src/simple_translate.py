from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def translate_en_vi_vinai(text):
    # Load the specific VinAI model for En -> Vi
    model_name = "vinai/vinai-translate-en2vi" 
    
    # Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="en_XX")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Prepare input
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    
    # Generate translation
    # beam_search usually gives better quality than greedy search
    output_ids = model.generate(
        input_ids,
        decoder_start_token_id=tokenizer.lang_code_to_id["vi_VN"],
        num_return_sequences=1,
        num_beams=5,
        max_length=1024
    )
    
    # Decode output
    translated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return translated_text

if __name__ == "__main__":
    text = "aboard"
    
    print("Loading model (this may take a moment first time)...")
    try:
        vi_text = translate_en_vi_vinai(text)
        print(f"English: {text}")
        print(f"Vietnamese: {vi_text}")
    except Exception as e:
        print(f"Error: {e}")
        print("Ensure you have installed: pip install transformers torch sentencepiece")
