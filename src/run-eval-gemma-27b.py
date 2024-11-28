import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import json
import re
import time

import os


torch.manual_seed(1112)


#from huggingface_hub import login
#login()



device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)



tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-27b-it", cache_dir="./gemma-2-27b-it-tokenizer")

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-27b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    cache_dir="./gemma-2-27b-it-model").eval()

print(tokenizer.chat_template)
print("\n")

############################################################



def remove_non_letter(s):
    return re.sub(r'[^a-zA-Z]', '', s)


def parse_output(s: str) -> int:
    # if true return 1 else return 0

    s = s.strip().split()

    for token in s:
        token = remove_non_letter(token).lower()
        if token == "yes":
            return 1
        if token == "no":
            return 0

    return 0



###########################################
out_file = open("./s5-new-results-gemma_27b-sample-01.txt", "a", encoding="utf8")

file_names = sorted( os.listdir("./data/s5/new-sample-01") )

acc_list = []


for fname in file_names:

    path = "./data/s5/new-sample-01/" + fname.strip()
    
    if not os.path.isfile(path):
        continue

    lines = open(path, "r", encoding="utf8").readlines()

    cot_flag = False

    if cot_flag:
        system_prompt = "You are an intelligent, helpful assistant. You need to find the pattern in the given list of examples, then you need to decide if the new example at the end belongs to yes or no. You should first reply with yes or no, then explain your reasoning step by step.\n"
        new_token_num = 512
    else:
        system_prompt = "You are an intelligent, helpful assistant. You need to find the pattern in the given list of examples, then you need to decide if the new example at the end belongs to yes or no. You should reply with only yes or no, and nothing else.\n"
        new_token_num = 1


    tp, tn, fp, fn = 0, 0, 0, 0
    count = 0

    start_time = time.time()
    for line in lines:
        count += 1

        line = json.loads(line)

        prompt = line["input"]
        gold_truth = 1 if "Yes." in line["label"] else 0


        messages = [
                {"role": "user", "content": system_prompt + prompt}
        ]

        #####
        input_ids = tokenizer.apply_chat_template(conversation=messages,
                                                  add_generation_prompt=True,
                                                  tokenize=True,
                                                  return_tensors='pt').to(model.device)

        output_ids = model.generate(input_ids,
                                    max_new_tokens=new_token_num,
                                    do_sample=False, num_beams=1,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.eos_token_id)

        llm_response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)


        #####


        ans = parse_output(llm_response)  # 1 if true else 0
        if count % 500 == 0:
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

    numbers = (tp, tn, fp, fn)

    out_file.write(fname + "\n")
    out_file.write(str(numbers) + "\n")


    pr = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # f1 = (2 * pr * recall) / (pr + recall)
    acc = (tp + tn) / (tp + tn + fp + fn)

    acc_list.append(acc)

    print(fname)
    print("numbers: " + str(numbers))
    print("acc: " + str(acc) + "\n")

    results = (pr, acc)

    out_file.write("(pr,  acc) \n")
    out_file.write(str(results) + " \n\n")



count = 0
for ac in acc_list:
    out_file.write(str(ac * 100) + "\n")

    count += 1
    if count % 18 == 0:
        out_file.write("\n")


out_file.close()
###########################################



