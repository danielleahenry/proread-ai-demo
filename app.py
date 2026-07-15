import gradio as gr
from transformers import T5ForConditionalGeneration, T5Tokenizer
import textstat

# load fine tuned model
model_name = "daniahenry/proread-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def simplify_text(complex_text):
    # 1. input prefixing
    input_text = "simplify: " + complex_text
    
    # 2. tokenization and encoding
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # 3 iterative decoding 
    outputs = model.generate(
        inputs,
        max_length=150,
        min_length=10,          
        num_beams=10,           
        temperature=0.9,        
        do_sample=True,        
        no_repeat_ngram_size=2, 
        early_stopping=True
    )
    
    # decode the numerical vectors back into text
    simplified_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 4. metric calculations (Grade Level, ASL, ASW)
    # originaltext metrics
    orig_flesch = textstat.flesch_kincaid_grade(complex_text)
    orig_words = max(1, textstat.lexicon_count(complex_text))
    orig_sentences = max(1, textstat.sentence_count(complex_text))
    orig_asl = orig_words / orig_sentences
    orig_asw = textstat.syllable_count(complex_text) / orig_words
    
    # simplified
    new_flesch = textstat.flesch_kincaid_grade(simplified_text)
    new_words = max(1, textstat.lexicon_count(simplified_text))
    new_sentences = max(1, textstat.sentence_count(simplified_text))
    new_asl = new_words / new_sentences
    new_asw = textstat.syllable_count(simplified_text) / new_words
    
    metrics_output = f"""
    📊 Original Text Metrics:
    - Flesch-Kincaid Grade: {orig_flesch}
    - Avg Sentence Length: {orig_asl:.1f} words
    - Avg Syllables/Word: {orig_asw:.1f}
    
    📉 Simplified Text Metrics:
    - Flesch-Kincaid Grade: {new_flesch}
    - Avg Sentence Length: {new_asl:.1f} words
    - Avg Syllables/Word: {new_asw:.1f}
    """
    
    return simplified_text, metrics_output

# 5. visual formatting
custom_css = """
@import url('https://fonts.cdnfonts.com/css/opendyslexic');

.dyslexic-text textarea {
    font-family: 'OpenDyslexic', sans-serif !important;
    text-align: left !important;
    line-height: 2.0 !important;
    letter-spacing: 0.1em !important;
}
"""

# 6. web dash
with gr.Blocks() as demo:
    gr.Markdown("# ProRead AI: Cognitive Load Reduction System")
    gr.Markdown("Enter complex text below to apply automated lexical and syntactic simplification.")
    
    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(label="Original Complex Text", lines=8)
            submit_btn = gr.Button("Simplify Text", variant="primary")
        with gr.Column():
            output_box = gr.Textbox(label="Simplified Output (Dyslexia-Friendly Format)", lines=8, elem_classes="dyslexic-text")
            metrics_box = gr.Textbox(label="Readability Validation", lines=8)
            
    # connect the button to the fxn
    submit_btn.click(fn=simplify_text, inputs=input_box, outputs=[output_box, metrics_box])

# launch
demo.launch(css=custom_css)