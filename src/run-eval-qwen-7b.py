import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
import json
import re
import time

import os
import argparse



# from huggingface_hub import login
# login()

torch.manual_seed(1112)

parser = argparse.ArgumentParser(prog='LLM concept learning')
parser.add_argument('--data_path', '-d', type=str, required=True, help='path to data folder')
args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)




model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    torch_dtype="auto",
    device_map = "balanced",
    cache_dir="./Qwen2-7B-Instruct-model").eval()



tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct", cache_dir="Qwen2-7B-Instruct-tokenizer")

print(tokenizer.chat_template)
print("\n")

############################################################



def remove_non_letter(s):
    return re.sub(r'[^a-zA-Z]', '', s)


def parse_output(s: str) -> int:
    # if true return 1 else return 0

    s = s.strip().split()

    for token in s[-5:]:
        token = remove_non_letter(token).lower()
        if token == "yes":
            return 1
        if token == "no":
            return 0

    return 0



########################################### '''
out_file = open("./qwen_7b-results.txt", "a", encoding="utf8")

file_names = sorted( os.listdir(args.data_path) )

for fname in file_names:
    path = args.data_path + fname.strip()

    if not os.path.isfile(path):
        continue

    cot_flag = False

    lines = open(path, "r", encoding="utf8").readlines()

    if cot_flag:
        system_prompt = "You are an intelligent, helpful assistant. You need to find the pattern in the given list of examples, then you need to decide if the new example at the end belongs to yes or no. You should explain your reasoning step by step. At the end of your response, reply with yes or no."
        new_token_num = 2048
    else:
        system_prompt = "You are an intelligent, helpful assistant. You need to find the pattern in the given list of examples, then you need to decide if the new example at the end belongs to yes or no. You should reply with only yes or no, and nothing else."
        new_token_num = 1


    # system_prompt = "You are an intelligent, helpful assistant. You need to determine if the given city is a national capital. You should reply with only yes or no, and nothing else."


    tp, tn, fp, fn = 0, 0, 0, 0
    count = 0

    start_time = time.time()
    for line in lines:
        count += 1

        line = json.loads(line)

        prompt = line["input"]
        gold_truth = 1 if "Yes" in line["label"] else 0


        messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
        ]


        #####
        '''text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ) 

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device) '''

        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt").to(model.device)

        # model_inputs.input_ids,
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=new_token_num,
            do_sample=False, num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        # zip(model_inputs.input_ids, generated_ids)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, generated_ids)
        ]

        llm_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        #####


        ans = parse_output(llm_response)  # 1 if true else 0
        if count % 1000 == 0:
            print("time: " + str(time.time() - start_time))
            print(count)
            print(llm_response)
            print("LM response: " + str(ans))
            print("gold: " + str(gold_truth) + "\n")


        if ans == gold_truth:
            if gold_truth == 1:
                tp += 1
            else:
                tn += 1
        else:
            if gold_truth == 1:
                fp += 1
            else:
                fn += 1



    pr = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # f1 = (2 * pr * recall) / (pr + recall)
    acc = (tp + tn) / (tp + tn + fp + fn)

    numbers = (tp, tn, fp, fn)

    print(fname)
    print(numbers)
    print("acc: " + str(acc) + "\n\n")



    out_file.write(fname + "\n")
    out_file.write(str(numbers) + "\n")

    results = (pr, acc)

    out_file.write("(pr,  acc) \n")
    out_file.write(str(results) + " \n\n")


out_file.close()


###########################################

del model
torch.cuda.empty_cache()


