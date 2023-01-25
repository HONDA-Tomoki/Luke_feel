from transformers import AutoTokenizer, LukeForQuestionAnswering
import torch
import json
MODEL_NAME='studio-ousia/luke-japanese-base-lite'
tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)
model=LukeForQuestionAnswering.from_pretrained(MODEL_NAME)


import os, json
dataset_dir = "DDQA-1.0/RC-QA"
list_file = ["DDQA-1.0_RC-QA_train.json", "DDQA-1.0_RC-QA_dev.json", "DDQA-1.0_RC-QA_test.json"]
list_dataset = []

for fil in list_file:
  with open(os.path.join(dataset_dir, fil),encoding='utf-8') as f:
    dataset = json.load(f)
    list_dataset.append(dataset['data'][0]['paragraphs'])
    print(len(dataset['data'][0]['paragraphs']))

list_train, list_valid, list_test = list_dataset

# cl-tohoku/bert-base-japanese-whole-word-maskingのモデルは最大512トークンまで対応しているが、
# 学習時のGPUメモリ消費を抑えるため256としている
n_token = 251

def is_in_span(idx, span):
  return span[0] <= idx and idx < span[1]


from collections import defaultdict

def preprocess(examples, is_test=False):
  dataset = defaultdict(list)
  all_starts, all_ends = [], []

  for example in examples:
    for qa in example["qas"]:
      context, question, answers = example["context"], qa["question"], qa["answers"]
      starts, ends = [], []

      for i,answer in enumerate(answers):
        encode = tokenizer(question, context)["input_ids"]
        tokenized = tokenizer.decode(encode)

        decode_str = tokenized.replace(" ", "").replace("<s>", "").replace("[PAD]", "").replace("##", "")
        
        # decode後のコンテクストの開始位置（質問文長）
        len_question = decode_str.find('</s></s>')

        cnt = 0
        start_position = 0
        for i_t,e in enumerate(encode):
          tok = tokenizer.decode(e).replace(" ", "")

          if tok == "</s>" or tok == "<s>" or tok == "[PAD]":
            continue
          else:
            if cnt <= len_question + answer["answer_start"]:
              start_position = i_t
            if cnt <= len_question + answer["answer_start"] + len(answer["text"]):
              end_position = i_t

          cnt += len(tok.replace("</s>", "").replace("<s>",""))
        
        starts.append(start_position)
        ends.append(end_position)

        if (not is_test) or (i == 0):
          dataset["contexts"].append(context)
          dataset["questions"].append(question)
          dataset["input_ids"].append(encode)
          dataset["tokenized"].append(tokenized)

          dataset["start_positions"].append(start_position)
          dataset["end_positions"].append(end_position)

      all_starts.append(starts)
      all_ends.append(ends)
  all_answers = (all_starts, all_ends)
  return dataset, all_answers


from torch.utils.data import Dataset, DataLoader

class QADataset(Dataset):
  def __init__(self, dataset, is_test=False):
    self.dataset = dataset
    self.is_test = is_test

  def __getitem__(self, idx):
    data = {'input_ids': torch.tensor(self.dataset["input_ids"][idx])}
    if not self.is_test:
      data["start_positions"] = torch.tensor(self.dataset["start_positions"][idx])
      data["end_positions"] = torch.tensor(self.dataset["end_positions"][idx])
    return data

  def __len__(self):
    return len(self.dataset["input_ids"])


from torch.utils.data import Dataset, DataLoader

class QADataset(Dataset):
  def __init__(self, dataset, is_test=False):
    self.dataset = dataset
    self.is_test = is_test

  def __getitem__(self, idx):
    data = {'input_ids': torch.tensor(self.dataset["input_ids"][idx])}
    if not self.is_test:
      data["start_positions"] = torch.tensor(self.dataset["start_positions"][idx])
      data["end_positions"] = torch.tensor(self.dataset["end_positions"][idx])
    return data

  def __len__(self):
    return len(self.dataset["input_ids"])


dataset_train = QADataset(preprocess(list_train)[0])
dataset_valid = QADataset(preprocess(list_valid)[0])
pp_test, test_answers = preprocess(list_test, is_test=True)
dataset_test = QADataset(pp_test, is_test=True)


from transformers import Trainer, TrainingArguments
training_config = TrainingArguments(
  output_dir = 'C://Users//tomot//desktop//Python//luke_squad_large',
  num_train_epochs = 3, 
  per_device_train_batch_size = 8,
  per_device_eval_batch_size = 8,
  warmup_steps = 500,
  weight_decay = 0.1,
  do_eval = True,
  save_steps = 470
)

trainer = Trainer(
    model = model,                         
    args = training_config,
    tokenizer = tokenizer,
    train_dataset = dataset_train,
    eval_dataset = dataset_valid
)

trainer.train()

torch.save(model, 'C:\\Users\\tomot\\desktop\\Python\\luke_squad_large\\My_luke_model_squad.pth')

result = trainer.predict(dataset_test)
import numpy as np
predictions = (np.argmax(result[0][0], axis=1), np.argmax(result[0][1], axis=1))

# トークン単位でのExact Match（厳密一致）とF1を計算

def evaluate(ground_truth, predictions):
  em, f1 = 0., 0.
  n_data = len(ground_truth[0])
  for answer_starts, answer_ends, pred_start, pred_end in zip(ground_truth[0], ground_truth[1], predictions[0], predictions[1]):
    for answer_start, answer_end in zip(answer_starts, answer_ends):
      if pred_start == answer_start and pred_end == answer_end: 
        em += 1
        break
    
    f1_candidate = [calc_f1(ps, pe, pred_start, pred_end) for ps, pe in zip(answer_starts, answer_ends)]
    f1 += max(f1_candidate)
  return {"em": (em / n_data), "f1": (f1 / n_data)}

def calc_f1(gt_start, gt_end, pred_start, pred_end):
  tp = max(0, (1 + min(gt_end, pred_end) - max(gt_start, pred_start)))
  precision = tp / (1 + pred_end - pred_start)  if 1 + pred_end - pred_start > 0 else 0
  # 通常、1 + gt_end - gt_start > 0がFalseになることはあり得ないが念のため
  recall = tp / (1 + gt_end - gt_start) if 1 + gt_end - gt_start > 0 else 0
  if precision * recall > 0:
    return 2 * (precision * recall) / (precision + recall)
  return 0.

emf1=str(evaluate(test_answers, predictions))

print(emf1)
print('finished')


