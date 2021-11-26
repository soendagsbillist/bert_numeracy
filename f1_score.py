from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import torch
from transformers import BertForQuestionAnswering


def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s)))

def create_segment_ids(input_ids):
  sep_index = input_ids.index(tokenizer.sep_token_id)
  question_segment = sep_index + 1
  passage_segment = len(input_ids) - question_segment
  segment_ids = [0] * question_segment + [1] * passage_segment
  return segment_ids


def test_accuracy(model, data_generator):
    precisions = []
    recalls = []
    # model.to(device)

    for index, data in enumerate(tqdm(data_generator)):
        for q in data:
            answ = q['answers_spans']
            que = ''.join(q['question'])
            passage = ''.join(q['passage'])
    
            if 'span' or 'date' in list(list(zip(*answer['types']))[0]):
                input_ids = tokenizer.encode(que,
                                        passage,
                                        max_length=384,
                                        truncation="only_second")
                segment_ids = create_segment_ids(input_ids)
    
                outputs = model(torch.tensor([input_ids]),
                         token_type_ids=torch.tensor([segment_ids]),
                         return_dict=True)
                predicted_start = torch.argmax(outputs.start_logits)
                predicted_end = torch.argmax(outputs.end_logits)
                # normalize predicted answer
                tokens = tokenizer.convert_ids_to_tokens(input_ids)
                normalized_prediction = normalize_answer(''.join(
                                        tokens[predicted_start:predicted_end+1]))
                # get ground truth tokens
                gold_tokens = ''.join(answ['spans'][0])

                # check for digits
                if normalized_prediction.strip().isdigit():
                    if normalized_prediction.strip() == gold_tokens.strip()
                        precision = 1
                        recall = 1
                    else:
                        precision = 0
                        recall = 0
                # general case 
                else:
                    common = Counter(normalized_prediction) & Counter(gold_tokens)
                    num_same = sum(common.values())
    
                # calculate precision, recall, f1
    
                if num_same == 0 or len(normalized_prediction) == 0:
                  precision = 0
                  recall = 0
                else:
                  precision = 1.0 * num_same / len(normalized_prediction)
                  recall = 1.0 * num_same / len(gold_tokens)
    
                precisions.append(precision)
                recalls.append(recall)

    avg_recall = sum(recalls) / len(recalls)
    avg_precision = sum(precisions) / len(precisions)
    f1 = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall)
    return f1

if __name__=="__main__":
    dataset = load_dataset('drop')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    validation_data = dataset['validation']
    val_generator = DataLoader(validation_data,
                                batch_size=8,
                                collate_fn=lambda x: x,
                                num_workers=2)
    print(f'F1 Score: {test_accuracy(model, val_generator)}')
