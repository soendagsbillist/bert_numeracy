from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BertTokenizer, BertForMaskedLM
import torch
import random

def convert_to_base(num: int, base: int, numerals="0123456789abcdefghijklmnopqrstuvwxyz") -> str:
    return ((num == 0) and numerals[0]) or (
        convert_to_base(num // base, base, numerals).lstrip(numerals[0]) + numerals[num % base])


def convert_to_character(number: str, separator: str, invert_number: bool, max_digits: int) -> str:
    if max_digits > 0:
        signal = None
        if number[0] == '-':
            signal = '-'
            number = number[1:]
        number = (max_digits - len(number)) * '0' + number
        if signal:
            number = signal + number
    if invert_number:
        number = number[::-1]
    return separator.join(number)


def convert_to_10based(number: str, invert_number: bool) -> str:
    signal = None
    if number[0] == '-':
        signal = '-'
        number = number[1:]

    output = []
    for i, digit in enumerate(number[::-1]):
        if i > 0:
            output.append('1' + i * '0')
        output.append(digit)

    if signal:
        output.append(signal)

    # The output is already inverted. If we want it to _not_ be inverted, then we invert it.
    if not invert_number:
        output = output[::-1]

    return ' '.join(output)


def convert_to_10ebased(number: str, split_type: str, invert_number: bool) -> str:
    signal = None
    if number[0] == '-':
        signal = '-'
        number = number[1:]

    output = []
    for i, digit in enumerate(number[::-1]):
        if split_type is None:
            output.append('10e' + str(i))
        elif split_type == 'underscore':
            output.append('10e' + '_'.join(str(i)))
        elif split_type == 'character':
            output.append(' '.join('D' + str(i) + 'E'))
        else:
            raise Exception(f'Wrong split_type: {split_type}')
        output.append(digit)

    if signal:
        output.append(signal)

    # The output is already inverted. If we want it to _not_ be inverted, then we invert it.
    if not invert_number:
        output = output[::-1]

    return ' '.join(output)

class NumeracyDataset(Dataset):
  def __init__(self, n_examples: int, min_digits: int, max_digits: int,
               operation: str, orthography: str, base_number: int,
               invert_question: bool, invert_answer: bool, balance: bool):
    self.operation = operation
    self.orthography = orthography
    self.invert_answer = invert_answer
    # what's invert_qustion?
    self.invert_question = invert_question
    self.base_number = base_number
    self.max_digits = max_digits
    self.examples = [(random.randint(0, int(max_digits * '9')),
                      random.randint(0, int(max_digits * '9')))
                      for _ in range(n_examples)]

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, idx):
    first_term, second_term = self.examples[idx]

    if self.operation == 'addition':
      operation_term = 'plus'
      result = first_term + second_term
    elif self.operation == 'subtraction':
      operation_term = 'minus'
      result = first_term - second_term
    else:
      raise Exception(f'Invalid operation: {self.operation}')

    first_term = self.convert_number(first_term,
                                     invert_number=self.invert_question)
    second_term = self.convert_number(second_term,
                                      invert_number=self.invert_question)
    answer = self.convert_number(result,
                                 invert_number=self.invert_question)

    #return f'What is {first_term} {operation_term} {second_term}', answer
    return f'{first_term} {operation_term} {second_term} equals {answer}'

  def convert_number(self, number: int, invert_number: bool) -> str:
    number = str(number)
    if self.base_number != 10:
      number = convert_to_base(num=int(number), base=self.base_number)

    if self.orthography == 'decimal':
      return convert_to_character(
          number=number, separator='', invert_number=invert_number,
          max_digits=-1)
    elif self.orthography == 'character':
      return convert_to_character(
          number=number, separator=' ', invert_number=invert_number,
          max_digits=-1)
    elif self.orthography == 'character_fixed':
      return convert_to_character(
          number=number, separator=' ', invert_number=invert_number,
          max_digits=sefl.max_digits)
    elif self.orthography == 'underscore':
      return convert_to_character(
          number=number, separator='_', invert_number=invert_number,
          max_digits=-1)
    elif self.orthography == 'words':
      return num2words(int(number))
    elif self.orthography == '10based':
      return convert_to_10based(number, invert_number=invert_number)
    elif self.orthography == '10ebased':
      return convert_to_10ebased(number, split_type=None,
                                   invert_number=invert_number)
    else:
      raise Exception(f'Wrong orthography: {self.orthography}')

train_size = 10000
min_digits_train = 2
max_digits_train = 3
operation = 'addition'
orthography = 'decimal'
base_number = 10
invert_question = False
invert_answer = False
balance_train = False

dataset_train = NumeracyDataset(n_examples=train_size, min_digits=min_digits_train,
                          max_digits=max_digits_train,
                          operation=operation, orthography=orthography,
                          base_number=base_number, invert_question=invert_question,
                          invert_answer=invert_answer, balance=balance_train)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_generator = DataLoader(dataset_train)

def convert_to_string(generator: DataLoader) -> str:
    equations = []
    for idx, item in enumerate(train_generator):
      itr += 1
      equation = ''.join(item)
      equations.append(equation)
    return equations

equations = convert_to_str(train_generator)
text = '\n'.join(equations)

def mask_data(text: str):
  inputs = tokenizer(text, return_tensors='pt')
  inputs['labels'] = inputs.input_ids.detach().clone()
  rand = torch.rand(inputs.input_ids.shape)
  mask_arr = rand < 0.15 * (inputs.input_ids != 101) * (inputs.input_ids != 102)
  mask_selection = torch.flatten(mask_arr[0].nonzero()).tolist()
  inputs.input_ids[0, mask_selection] = 103
  return inputs

def encode(data: NumeracyDataset, tokenizer: BertTokenizer):
  selection = []
  inputs = tokenizer(list(data), return_tensors='pt', padding='longest')
  # copy input_ids and append as labels
  inputs['labels'] = inputs.input_ids.detach().clone()
  # mask 15%
  rand = torch.rand(inputs.input_ids.shape)
  # crate a binary array and define what to [MASK]
  # tokens: 101(beginning), 102(end), 0(padding),
  # 19635(equals), 4606(plus)
  mask_arr = (rand < 0.15 * (inputs.input_ids != 101) *
              (inputs.input_ids != 102) *
              (inputs.input_ids != 0) *
              (inputs.input_ids != 19635) *
              (inputs.input_ids != 4606))
  # populate selection list with selections, duh.
  for i in range(0, mask_arr.shape[0]):
    mask_selection = torch.flatten(mask_arr[i].nonzero()).tolist()
    selection.append(mask_selection)
  # [MASK]
  for i in range(0, mask_arr.shape[0]):
    inputs.input_ids[i, selection[i]] = 103
  return inputs

def validate(model: BertForMaskedLM, epochs, test_data_generator: DataLoader, EM: bool):
    loop = tqdm(test_data_generator, leave=True)
    model.eval()
    model.to(device)
    total_train_loss = 0
    for batch in loop:
          with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids,
                                attention_mask=attention_mask,
                                labels=labels)

                outputs["input_ids"] = input_ids[0]
                masked_ids = []
                preds = []
                expected = []
                em = []
                for i in range(input_ids.shape[0]):
                    masked_id = torch.nonzero(input_ids[i].view(-1) == tokenizer.mask_token_id, as_tuple=False)
                    masked_ids.append(masked_id)
                    output_logits = outputs["logits"]
                for i in range(output_logits.shape[0]):
                    logits = output_logits[i, masked_ids[i], :]
                    ground_truth = labels[i, masked_ids[i]]
                    if logits.shape[0] > 1:
                        for i in (range(logits.shape[0])):
                            y_hat = torch.argmax(logits[i])
                            y = ground_truth[i].item()
                            if EM == False and postprocess_subwords(tokenizer.decode(y_hat)).isdigit():
                                preds.append(int(postprocess_subwords(tokenizer.decode(y_hat))))
                                expected.append(int(postprocess_subwords(tokenizer.decode(y))))
                            elif EM == True:
                            # compare tokens
                                if int(y_hat) == int(y):
                                    em.append(1)
                                else:
                                    em.append(0)

                    elif logits.shape[0] == 1:
                        y_hat = torch.argmax(logits)
                        y = ground_truth.item()
                        if EM == False and postprocess_subwords(tokenizer.decode(y_hat)).isdigit():
                            preds.append(int(postprocess_subwords(tokenizer.decode(y_hat))))
                            expected.append(int(postprocess_subwords(tokenizer.decode(y))))
                        elif EM == True:
                            if int(y_hat) == int(y):
                                em.append(1)
                            else:
                                em.append(0)

                if EM == True:
                    exact_match = sum(em) / len(em)
                    loop.set_description(f'Evaluation with EM: ')
                    loop.set_postfix(EM=exact_match)
                    total_train_loss += exact_match
                else:
                    rmse = RMSELoss(preds, expected)
                    loop.set_description(f'Evaluation with RMSE: ')
                    loop.set_postfix(error=rmse.item())
                    total_train_loss += rmse
    avg_train_loss = total_train_loss / len(test_data_generator)
    print("  Average validation loss: {0:.2f}".format(avg_train_loss))
