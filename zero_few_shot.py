import os
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd
from openai import OpenAI, InternalServerError

from tqdm.auto import tqdm

tqdm.pandas()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LANGUAGE_DICT = {
    "zh": "Chinese",
    "es": "Spanish",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "uk": "Ukrainian",
    "de": "German",
    "am": "Amharic",
    "en": "English"
}

def get_examples() -> dict:
    diectory = "detoxified"
    examples = {}
    for file_name in os.listdir(diectory):
        df = pd.read_csv(os.path.join(diectory, file_name), sep="\t")
        language = file_name.split(".")[0]
        examples[language] = df
    return examples

def gpt_generation(query: str, client: OpenAI) -> str:
    completion = client.chat.completions.create(model="gpt-3.5-turbo",
                                                messages=[{"role": "user", "content": query}],
                                                temperature=0.2,
                                                top_p=0.95)
    return completion.choices[0].message.content

def model_generation(model_tag,
                     prompt_English,
                     shots,
                     toxic,
                     language,
                     client,
                     tokenizer,
                     model):
    language_prompts = {"Ukrainian": "Перепишіть цей текст так, щоб він не звучав образливо, i його значення та мова джерела залишалися тими ж: TOXIC Переписаний текст:",
                        "Arabic": "أعد كتابة هذا النص بحيث لا يبدو مسيئًا وأن معناه ولغة المصدر تظل كما هي: TOXIC النص المعاد كتابته:",
                        "German": "Schreiben Sie diesen Text so um, dass er nicht beleidigend klingt und seine Bedeutung sowie die Ausgangssprache gleich bleiben: TOXIC Umgewschriebener Text:",
                        "Hindi": "इस पाठ को पुनः लिखें ताकि यह आक्षेपणीय न लगे और इसका अर्थ व स्रोत भाषा समान रहे: TOXIC पुनः लिखित पाठ:",
                        "Chinese": "重写这段文字, 使其听起来不冒犯, 其意义和来源语言保持不变: TOXIC 重写后的文字：",
                        "Spanish": "Reescribe este texto de manera que no suene ofensivo y su significado y el idioma de origen permanezcan iguales: TOXIC Texto reescrito:",
                        "Amharic": "ይህንን ጽሑፍ እንዲህ እንዳይመስል እና ማንቴው እና የምንጭ ቋንቋው እንደገና እኩል ይሁኑ፡፡ TOXIC ዳግም የተጻፈ ጽሑፍ፡",
                        "Russian": "Перепишите этот текст так, чтобы он не звучал оскорбительно, и его значение и исходный язык остались теми же: TOXIC Переписанный текст:",
                        "English": "Rewrite this text so that it does not sound offensive and its meaning and source language stay the same. EXAMPLES TOXIC Rewritten text:"}
    if prompt_English:
        prompt = language_prompts["English"]
    else:
        prompt = language_prompts[language]
    toxic = toxic.replace('"', '\"')
    if shots:
        shots = shots[language].head(5).to_dict(orient="records")
        shots4prompt = "\n" + "\n".join([f"{shot['toxic']} Rewritten text: {shot['neutral']}" for shot in shots]) + "\n"
        current_prompt = prompt.replace("EXAMPLES", shots4prompt)
    else:
        current_prompt = prompt.replace("EXAMPLES", "")
    current_prompt = current_prompt.replace("TOXIC", toxic)
    if "gpt-3.5-turbo" == model_tag:
        try:
            detoxified = gpt_generation(current_prompt, client)
        except InternalServerError:
            detoxified = toxic
    else:
        inputs = tokenizer.encode(current_prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(inputs,
                                 max_new_tokens=256,
                                 do_sample=True,
                                 temperature=0.4,
                                 top_p=0.95)
        detoxified = tokenizer.decode(outputs[0])
        detoxified = detoxified.replace(current_prompt, "").strip()
        detoxified = detoxified.replace("</s>", "")
        detoxified = detoxified.replace("<pad> ", "")
    return detoxified

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate text generation outputs.")
    parser.add_argument("--filename", type=str, required=True, help="The JSONL file to evaluate.")
    parser.add_argument("--model_tag", type=str, required=True, help="The model used to generate the outputs: tag from OpenAI or Hugging Face.")
    parser.add_argument("--prompt_English", type=bool, default=True, help="Whether the prompt is in English or not.")
    parser.add_argument("--shots", type=bool, default=False, help="Whether to include examples in the prompt.")
    parser.add_argument("--api_key", type=str, help="The OpenAI API key.")
    args = parser.parse_args()

    if args.model_tag == "bigscience/bloomz-7b1":
        tokenizer = AutoTokenizer.from_pretrained(args.model_tag)
        model = AutoModelForCausalLM.from_pretrained(args.model_tag, 
                                                     torch_dtype="auto", 
                                                     device_map=DEVICE)
        client = None
        api_key = None
    elif args.model_tag == "bigscience/mt0-xxl":
        tokenizer = AutoTokenizer.from_pretrained(args.model_tag)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_tag, 
                                                      load_in_8bit=True, 
                                                      device_map=DEVICE)
        client = None
        api_key = None
    elif "gpt-3.5-turbo" == args.model_tag:
        tokenizer = None
        model = None
        client = OpenAI(api_key=args.api_key)
    else:
        raise AssertionError("Model tag not recognized.")

    if args.shots:
        shots = get_examples()
    data = pd.read_csv(args.filename, sep="\t")
    data["neutral_sentence"] = data.progress_apply(lambda row: model_generation(args.model_tag,
                                                                            args.prompt_English,
                                                                            shots,
                                                                            row["toxic_sentence"],
                                                                            LANGUAGE_DICT[row["lang"]],
                                                                            client,
                                                                            tokenizer,
                                                                            model), axis=1)                             
    data["neutral_sentence"] = data["neutral_sentence"].str.replace("\n", " ")
    out_file = args.filename.split(".")[0] + "_detoxified.tsv"
    data.to_csv(out_file, index=False, sep="\t")
