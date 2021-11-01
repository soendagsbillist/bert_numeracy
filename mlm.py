from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
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
