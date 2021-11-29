from datasets import load_dataset
from transformers import BertForQuestionAnswering
from transformers import DistilBertTokenizerFast
from transformers import AdamW
from tqdm.auto import tqdm
import torch
import pathlib 

def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_tokens = "".join(answer['text'][0])
        start_idx = answer['answer_start'][0]
        end_idx = start_idx + len(gold_tokens)

        if context[start_idx:end_idx] == gold_tokens:
            answer['answer_end'] = end_idx
        else:
            for n in [1, 2]:
                if context[start_idx-n:end_idx-n] == gold_tokens:
                    answer['answer_start'] = start_idx - n
                    answer['answer_end'] = end_idx - n

def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start'][0]))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        shift = 1
        while end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - shift)
            shift += 1
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

def train(model, optimizer, data_loader, epoch):
    model.train()
    loop = tqdm(data_loader, leave=True)
    total_train_loss = 0
    for batch in loop:
        optimizer.zero_grad()
        #input_ids = batch['input_ids'].to(device)
        #attention_mask = batch['attention_mask'].to(device)
        #start_positions = batch['start_positions'].to(device)
        #end_positions = batch['end_positions'].to(device)
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        start_positions = batch['start_positions']
        end_positions = batch['end_positions']
        outputs = model(input_ids, attention_mask=attention_mask,
                        start_positions=start_positions,
                        end_positions=end_positions)

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(data_loader)
    print("Average training loss: {0:.2f}".format(avg_train_loss))

def validate(model, data_loader, epoch):
    model.eval()
    loop = tqdm(data_loader, leave=True)
    total_eval_loss = 0

    for batch in loop:
        #input_ids = batch['input_ids'].to(device)
        #attention_mask = batch['attention_mask'].to(device)
        #start_positions = batch['start_positions'].to(device)
        #end_positions = batch['end_positions'].to(device)
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        start_positions = batch['start_positions']
        end_positions = batch['end_positions']
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions)
            loss = outputs.loss
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
            total_eval_loss += loss.item()
    avg_eval_loss = total_eval_loss / len(data_loader)
    print("Average validation loss: {0:.2f}".format(avg_eval_loss))

if __name__=="__main__":
    PATH = pathlib.Path().resolve()
    num_epoch = 1
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = BertForQuestionAnswering.from_pretrained('./pretrained/')
    optimizer =  AdamW(model.parameters(), lr=5e-5)
    dataset = load_dataset('squad')
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    #train_answers = [dataset['train'][i]['answers'] for i in range(len(dataset['train']))]
    #train_contexts = [dataset['train'][i]['context'] for i in range(len(dataset['train']))]
    #train_questions = [dataset['train'][i]['question'] for i in range(len(dataset['train']))]
    #add_end_idx(train_answers, train_contexts)

    val_answers = [dataset['validation'][i]['answers'] for i in range(len(dataset['validation']))]
    val_contexts = [dataset['validation'][i]['context'] for i in range(len(dataset['validation']))]
    val_questions = [dataset['validation'][i]['question'] for i in range(len(dataset['validation']))]
    add_end_idx(val_answers, val_contexts)

    #train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
    #add_token_positions(train_encodings, train_answers)
    val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)
    add_token_positions(val_encodings, val_answers)

    #train_dataset = SquadDataset(train_encodings)
    val_dataset = SquadDataset(val_encodings)

    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True)

    for epoch in range(num_epoch):
        #train(model, optimizer, train_loader, epoch)
        validate(model, val_loader, epoch)

    model.save_pretrained(PATH) 
