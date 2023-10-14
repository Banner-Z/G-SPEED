import json
from typing import Dict, List
import argparse
import utils
import os
import random
import nltk

SEQ_DELIMETERS = {"tokens": " ",
                "labels": "SEPL|||SEPR",
                "operations": "SEPL__SEPR"}
START_TOKEN = "$START"
# MASK_TOKEN = "[MASK]"
PAD_TOKEN = "[PAD]"
DELETE_START_TOKEN = "[DELETE]"
DELETE_END_TOKEN = "[/DELETE]"
TRANSFORM_VERB_START_TOKEN = "[TRANSFORM_VERB]"
TRANSFORM_VERB_END_TOKEN = "[/TRANSFORM_VERB]"

def write2json(records, output_dir):
    with open(output_dir, "w", encoding="utf8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

def json2list(filename):
    # if the file is huge, this method is not efficient for memory.
    data = []
    with open(filename, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            data.append(json.loads(line))
    return data

def txt2list(filename):
    # if the file is huge, this method is not efficient for memory.
    data = []
    with open(filename, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.replace('\n', '')
            # line.replace('\r', '')
            data.append(line.split(' '))
    return data

def prepare_for_inference(tokens_and_tags, remain_delete=True, num_of_slot_tokens=4):
    records = []  
    # d=0
    for tokens, tags in tokens_and_tags:
        tokens = tokens['tokens']
        tags = tags
        # print(tokens)
        # print(tags)
        # raise 'edn'
        inputs = []
        masks = []
        i = 0
        for index,t in enumerate(tags):
            if t.startswith('B-'):
                tags[index] = t[2:]
        # d+=1
        # print(tags)
        # if d==2:
            # raise 'end'
        # all_tags = []
        # for t in tags:
        #     all_tags.extend(t)
        # all_tags = ''.join(all_tags)
        # print(all_tags)

        # if "$REPLACE" not in all_tags and "$APPEND" not in all_tags:
            # print(11)
            # continue

        # tags_copy = []
        # for t in tags:
        #     tags_copy.append(t.copy())
        
        while i < len(tags):
            # print(i)
            # if i >= len(tokens):
            #     break
            if tags[i] == '$KEEP':
                inputs.append(tokens[i])
                i += 1
                masks.append(0)
            elif tags[i] == '$DELETE':
                if remain_delete:
                    inputs.append(DELETE_START_TOKEN)
                    masks.append(0)
                    inputs.append(tokens[i])
                    masks.append(0)
                    inputs.append(DELETE_END_TOKEN)
                    masks.append(0)
                i += 1
            elif "$TRANSFORM" in tags[i]:
                if tags[i].startswith('$TRANSFORM_VERB-'):
                    inputs.append(TRANSFORM_VERB_START_TOKEN)
                    masks.append(0)
                    inputs.append(tokens[i])
                    masks.append(0)
                    inputs.append(TRANSFORM_VERB_END_TOKEN)
                    masks.append(0)
                    new_tokens = '[MASK]'
                    i += 1
                    # new_tokens.extend([PAD_TOKEN for _ in range(INSERT_LENTH-new_tokens_lenth)])
                    for _ in range(num_of_slot_tokens):
                        inputs.append(new_tokens)
                        masks.append(1)
                else:
                    new_tokens = utils.apply_reverse_transformation(tokens[i], tags[i])
                    if new_tokens[0] is not None:
                        # print(new_tokens)
                        inputs.extend(new_tokens)
                        masks.extend([0 for _ in range(len(new_tokens))])
                        i += 1
                    else:
                        # print(111)
                        if remain_delete:
                            inputs.append(DELETE_START_TOKEN)
                            masks.append(0)
                            inputs.append(tokens[i])
                            masks.append(0)
                            inputs.append(DELETE_END_TOKEN)
                            masks.append(0)
                        new_tokens = '[MASK]'
                        i += 1
                        for _ in range(num_of_slot_tokens):
                            inputs.append(new_tokens)
                            masks.append(1)
            elif "$MERGE_HYPHEN" in tags[i]:
                new_tokens = []
                while "$MERGE_HYPHEN" in ''.join(tags[i]):
                    # tags[i].remove("$MERGE_HYPHEN")
                    new_tokens.append(tokens[i])
                    i += 1
                inputs.append('-'.join(new_tokens))
                masks.append(0)
            elif "$MERGE_SPACE" in tags[i]:
                new_tokens = []
                while "$MERGE_SPACE" in ''.join(tags[i]):
                    # tags[i].remove("$MERGE_SPACE")
                    new_tokens.append(tokens[i])
                    i += 1
                inputs.append(''.join(new_tokens))
                masks.append(0)
            elif "$REPLACE" in tags[i]:
                if remain_delete:
                    inputs.append(DELETE_START_TOKEN)
                    masks.append(0)
                    inputs.append(tokens[i])
                    masks.append(0)
                    inputs.append(DELETE_END_TOKEN)
                    masks.append(0)
                new_tokens = '[MASK]'
                i += 1
                # new_tokens.extend([PAD_TOKEN for _ in range(INSERT_LENTH-new_tokens_lenth)])
                for _ in range(num_of_slot_tokens):
                    inputs.append(new_tokens)
                    masks.append(1)
                # i += 1
            elif "$APPEND" in tags[i]:
                inputs.append(tokens[i])
                masks.append(0)
                new_tokens = '[MASK]'
                # new_tokens.extend([PAD_TOKEN for _ in range(INSERT_LENTH-new_tokens_lenth)])
                for _ in range(num_of_slot_tokens):
                    inputs.append(new_tokens)
                    masks.append(1)
                i += 1
            else:
                print(tags[i])
        record = {
                "tokens": inputs,
                "mask": masks,
            }
        records.append(record)
    return records

def gector2json(input_file, output_file, mode, pred=None, num_of_slot_tokens=None):

    def _read(file_path, one_tag_only, generation):
        records = []
        with open(file_path, "r") as data_file:
            # kk=0
            print("Reading instances from lines in file at: ", file_path)
            for line in data_file:
                # kk+=1
                # if kk>20000 and kk<30000:
                #     continue
                line = line.strip("\n")
                # skip blank lines, and if you need to convert Lang8, take care of broken lines.(gector: datareader.py)
                if not line:
                    continue
                tokens_and_tags = [pair.rsplit(SEQ_DELIMETERS['labels'], 1)
                                   for pair in line.split(SEQ_DELIMETERS['tokens'])]
                try:
                    tokens = [token for token, tag in tokens_and_tags]
                    # handle the tags
                    if not generation:
                        tags = [tag if tag.split('_')[0] not in ['$APPEND','$REPLACE'] else tag.split('_')[0] for token, tag in tokens_and_tags]
                    else:
                        tags = [tag for token, tag in tokens_and_tags]
                    tags = [x.split(SEQ_DELIMETERS['operations']) for x in tags]
                    if one_tag_only:
                        tags = [x[0] for x in tags]     # take the first tag when there are more than one tags.
                except ValueError:
                    tokens = [token[0] for token in tokens_and_tags]
                    tags = None
                    print('ValueError!')
                
                if tokens and tokens[0] != START_TOKEN:
                    tokens = [START_TOKEN] + tokens
                
                record = {
                    "tokens": tokens,
                    "action_tags": tags,
                }
                records.append(record)
        return records
    
    def _prepare_for_generation(tokens_and_tags, remain_delete=True):
        records = []  
        # d=0
        for line in tokens_and_tags:
            tokens = line['tokens']
            tags = line['action_tags']
            inputs = []
            masks = []
            i = 0
            # d+=1
            # print(tags)
            # if d==2:
                # raise 'end'
            all_tags = []
            for t in tags:
                all_tags.extend(t)
            all_tags = ''.join(all_tags)
            # print(all_tags)

            if "$REPLACE" not in all_tags and "$APPEND" not in all_tags:
                # print(11)
                continue

            tags_copy = []
            for t in tags:
                tags_copy.append(t.copy())
            
            while i < len(tags):
                if len(tags[i]) == 0:
                    i += 1
                    continue
                if tags[i][0] == '$KEEP':
                    inputs.append(tokens[i])
                    i += 1
                    masks.append(0)
                elif tags[i][0] == '$DELETE':
                    if remain_delete:
                        inputs.append(DELETE_START_TOKEN)
                        masks.append(0)
                        while i<len(tags) and tags[i][0] == '$DELETE':
                            inputs.append(tokens[i])
                            masks.append(0)
                            i += 1
                        inputs.append(DELETE_END_TOKEN)
                        masks.append(0)
                    else:
                        i += 1
                elif "$TRANSFORM" in ''.join(tags[i]):
                    for index, tag in enumerate(tags[i]):
                        tansform_times = 0
                        if tag.startswith("$TRANSFORM"):
                            if tag.startswith('$TRANSFORM_VERB-'):
                                inputs.append(TRANSFORM_VERB_START_TOKEN)
                                masks.append(0)
                                inputs.append(tokens[i])
                                masks.append(0)
                                inputs.append(TRANSFORM_VERB_END_TOKEN)
                                masks.append(0)
                                new_tokens = tag.replace('$TRANSFORM_VERB-', '')
                                # new_tokens.extend([PAD_TOKEN for _ in range(INSERT_LENTH-new_tokens_lenth)])
                                inputs.append(new_tokens)
                                masks.append(1)
                                tags[i].pop(index)
                            else:
                                tansform_times += 1
                                new_tokens = utils.apply_reverse_transformation(tokens[i], tag)
                                inputs.extend(new_tokens)
                                masks.extend([0 for _ in range(len(new_tokens))])
                                tags[i].pop(index)
                    if tansform_times > 1:
                        print("The token: ", tokens[i], " has transformed more than one time, whitch may need your attention.\n", \
                                "The whole sentence is: ", ' '.join(tokens), "\nThe tags are: ", str(tags))
                elif "$MERGE_HYPHEN" in ''.join(tags[i]):
                    new_tokens = []
                    while "$MERGE_HYPHEN" in ''.join(tags[i]):
                        tags[i].remove("$MERGE_HYPHEN")
                        new_tokens.append(tokens[i])
                        i += 1
                    inputs.append('-'.join(new_tokens))
                    masks.append(0)
                elif "$MERGE_SPACE" in ''.join(tags[i]):
                    new_tokens = []
                    while "$MERGE_SPACE" in ''.join(tags[i]):
                        tags[i].remove("$MERGE_SPACE")
                        new_tokens.append(tokens[i])
                        i += 1
                    inputs.append(''.join(new_tokens))
                    masks.append(0)
                elif "$REPLACE" in ''.join(tags[i]):
                    new_tokens = ''
                    for index, tag in enumerate(tags[i]):
                        if tag.startswith("$REPLACE"):
                            if remain_delete:
                                inputs.append(DELETE_START_TOKEN)
                                masks.append(0)
                                inputs.append(tokens[i])
                                masks.append(0)
                                inputs.append(DELETE_END_TOKEN)
                                masks.append(0)
                            new_tokens = tag.split('_')[1].split('**?*')
                            # new_tokens = tag.split('_')[1:]

                            tags[i].pop(index)
                            break
                    new_tokens_lenth = len(new_tokens)
                    # new_tokens.extend([PAD_TOKEN for _ in range(INSERT_LENTH-new_tokens_lenth)])
                    inputs.extend(new_tokens)
                    masks.extend([1 for _ in range(len(new_tokens))])
                    # i += 1
                elif "$APPEND" in ''.join(tags[i]):
                    for index, tag in enumerate(tags_copy[i]):
                        if index == 0 and tag.startswith("$APPEND"):
                            inputs.append(tokens[i])
                            masks.append(0)
                        new_tokens = ''
                        if tag.startswith("$APPEND"):
                            new_tokens = tag.split('_')[1:]
                            new_tokens_lenth = len(new_tokens)
                            # new_tokens.extend([PAD_TOKEN for _ in range(INSERT_LENTH-new_tokens_lenth)])
                            inputs.extend(new_tokens)
                            masks.extend([1 for _ in range(len(new_tokens))])
                    i += 1
                else:
                    print(tags[i])
            record = {
                    "tokens": inputs,
                    "mask": masks,
                }
            records.append(record)
        return records

    if mode == 'tagging':
        records = _read(input_file, one_tag_only=True, generation=False)
        # train = records[:-5000]
        # valid = records[-5000:]
        train = []
        valid = []
        for r in records:
            if random.random() > 0.1:
                train.append(r)
            else:
                valid.append(r)
        write2json(records, output_file)
        write2json(valid, output_file.replace('train', 'valid'))
    elif mode == 'generation':
        records = _read(input_file, one_tag_only=False, generation=True)
        
        print("Reading finished.")
        records = _prepare_for_generation(records, remain_delete=True)
        train = []
        valid = []
        # train = records[:-5000]
        # valid = records[-5000:]
        for r in records:
            if random.random() > 0.1:
                train.append(r)
            else:
                valid.append(r)
        write2json(valid, output_file.replace('train', 'valid'))
        write2json(train, output_file)
    elif mode == 'inference':
        texts = _read(input_file, one_tag_only=False, generation=True)
        tags = txt2list(pred)
        print("Reading finished.")
        records = prepare_for_inference(zip(texts, tags), remain_delete=True, num_of_slot_tokens=num_of_slot_tokens)
        write2json(records, output_file)

def editeval2json(input_file, output_file, mode, pred=None, num_of_slot_tokens=None, split='nltk'):

    SEQ_DELIMETERS = {"tokens": " ",
                  "labels": "SEPL|||SEPR",
                  "operations": "SEPL__SEPR"}
    START_TOKEN = "$START"
    # MASK_TOKEN = "[MASK]"
    PAD_TOKEN = "[PAD]"
    DELETE_START_TOKEN = "[DELETE]"
    DELETE_END_TOKEN = "[/DELETE]"
    # INSERT_LENTH = 2

    if mode == 'tagging':
        records = []
        extension = input_file.split('.')[-1]
        if extension =='jsonl' or extension == 'json:':
            inputs = json2list(input_file)
            column = 'input' if 'input' in inputs[0].keys() else 'output'
        else:
            inputs = txt2list(input_file)

        for line in inputs:
            tokens = [START_TOKEN]
            if split == 'nltk':
                if extension =='jsonl' or extension == 'json:':
                    tokens.extend(nltk.word_tokenize(line[column]))
                    if 'task_type' in line.keys() and line['task_type'] == 'update':
                        pass
                else:
                    tokens.extend(nltk.word_tokenize(' '.join(line)))

            elif split == 'space':
                tokens = line.split()
            else:
                pass
            tags = ['$KEEP' for _ in range(len(tokens))]
            record = {
                'tokens': tokens,
                'action_tags': tags,
            }
            records.append(record)
        write2json(records, output_file)
    elif mode == 'generation':
        inputs = json2list(input_file)
        tags = txt2list(pred)
        records = prepare_for_inference(zip(inputs, tags), remain_delete=True, num_of_slot_tokens=num_of_slot_tokens)
        write2json(records, output_file)
    elif mode == 'results2json':
        inputs = json2list(input_file)
        records = []
        with open(pred, 'r') as fin:
            lines = fin.readlines()
            for index, line in enumerate(lines):
                # line = line[0].upper() + line[1:]
                # record = {
                #     'id': inputs[index]['id'],
                #     'output': line.replace('\n', '').replace('` ` ', "\"").replace('` `', "\"").replace("' '", "\"").replace(' ` ', "'").replace('`', "'") \
                #         .replace('( ', "(").replace(' )', ")").replace(' - ', "-").replace("''","\"")
                # }
                record = {
                    'id': inputs[index]['id'],
                    'output': line.replace('\n', '')
                }
                records.append(record)
        write2json(records, output_file)
        
def main(args):
    if args.func == 'gector2json':
        if args.mode == None:
            raise Exception('The argument \'mode\' is needed. Please choose \'tagging\' or \'generation\'.')
        gector2json(args.input_file, args.output_file, args.mode, args.pred, args.num_of_slot_tokens)
    elif args.func == 'editeval':
        if args.mode == None:
            raise Exception('The argument \'mode\' is needed. Please choose \'tagging\' or \'generation\'.')
        editeval2json(args.input_file, args.output_file, args.mode, args.pred, args.num_of_slot_tokens)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file',
                        help='Path to the input file',
                        required=True)
    parser.add_argument('-o', '--output_file',
                        help='Path to the output file',
                        required=True)
    parser.add_argument('--func',
                        type=str,
                        help='function name',
                        default='gector2json')
    parser.add_argument('--mode',
                        type=str,
                        help='mode name',
                        default=None)
    parser.add_argument('--pred',
                        type=str,
                        help='mode name',
                        default=None)
    parser.add_argument('--num_of_slot_tokens',
                        type=int,
                        help='',
                        default=1)
    parser.add_argument('--split',
                        type=str,
                        help='split method',
                        default='nltk')
    args = parser.parse_args()
    main(args)
