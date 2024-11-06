from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Chargement du tokenizer et du modèle pour la génération de texte
tokenizer = GPT2Tokenizer.from_pretrained("antoiloui/belgpt2")
model = GPT2LMHeadModel.from_pretrained("antoiloui/belgpt2")

# Préparation des entrées pour la génération de texte, avec attention_mask
inputs = tokenizer("Le Challenge #30daysgenerativeai c'est", return_tensors="pt")
attention_mask = inputs['attention_mask']

# Génération de texte en français en fournissant l'attention_mask et en définissant pad_token_id si nécessaire
text_generation = model.generate(
    input_ids=inputs['input_ids'],
    attention_mask=attention_mask,
    max_length=200,
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id  # Définir si nécessaire
)

# Affichage du texte généré
print(tokenizer.decode(text_generation[0], skip_special_tokens=True))
