import torch
from transformers import MLukeTokenizer,pipeline, LukeForQuestionAnswering

tokenizer = MLukeTokenizer.from_pretrained('studio-ousia/luke-japanese-base-lite')
model=LukeForQuestionAnswering.from_pretrained('Mizuiro-sakura/luke-japanese-base-finetuned-QA') # 学習済みモデルの読み込み

text={
    'context':'私の名前はEIMIです。好きな食べ物は苺です。 趣味は皆さんと会話することです。1987年ネバーランド寺院が建設された',
    'question' :'ネバーランド寺院が建設されたのはいつですか'
}

input_ids=tokenizer.encode(text['question'],text['context']) # tokenizerで形態素解析しつつコードに変換する
con=tokenizer.encode(text['question'])
output= model(torch.tensor([input_ids])) # 学習済みモデルを用いて解析
prediction = tokenizer.decode(input_ids[torch.argmax(output.start_logits): torch.argmax(output.end_logits)]) # 答えに該当する部分を抜き取る
prediction=prediction.replace('</s>','')
print(prediction)

#qa=pipeline('question-answering', model=model, tokenizer=tokenizer)


#if __name__=='__main__':
 #   result=qa(text)

  #  print(result)