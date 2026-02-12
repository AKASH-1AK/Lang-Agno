print("🚀 Translation script started")

from transformers import MarianMTModel, MarianTokenizer

# Public models (no authentication required)
HI_TO_EN = "Helsinki-NLP/opus-mt-hi-en"
EN_TO_HI = "Helsinki-NLP/opus-mt-en-hi"

print("🔄 Loading Hindi ↔ English models...")

hi_en_tokenizer = MarianTokenizer.from_pretrained(HI_TO_EN)
hi_en_model = MarianMTModel.from_pretrained(HI_TO_EN)

en_hi_tokenizer = MarianTokenizer.from_pretrained(EN_TO_HI)
en_hi_model = MarianMTModel.from_pretrained(EN_TO_HI)

print("✅ Models loaded successfully")


def translate_hi_to_en(text):
    inputs = hi_en_tokenizer(text, return_tensors="pt", padding=True)
    output = hi_en_model.generate(**inputs)
    return hi_en_tokenizer.decode(output[0], skip_special_tokens=True)


def translate_en_to_hi(text):
    inputs = en_hi_tokenizer(text, return_tensors="pt", padding=True)
    output = en_hi_model.generate(**inputs)
    return en_hi_tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    print("\n🌍 Translation Test (Hindi ↔ English)")
    user_input = input("Enter Hindi text: ")

    english_text = translate_hi_to_en(user_input)
    print("➡ English:", english_text)

    hindi_back = translate_en_to_hi(english_text)
    print("➡ Back to Hindi:", hindi_back)
