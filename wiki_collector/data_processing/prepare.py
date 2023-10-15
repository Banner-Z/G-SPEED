import argparse
import json
import os
from tqdm import tqdm
import glob
import nltk
import re
from difflib import SequenceMatcher
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

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

def list2txt(data, source, target):
    # if the file is huge, this method is not efficient for memory.
    with open(source, "w") as sf, open(target, "w") as tf:
        for line in data:
            sf.write(line['src_token']+'\n')
            tf.write(line['tgt_token']+'\n')
    print(len(data))

def list2jsonl(data, output_file):
    records = []
    for line in data:
        record = {
            'id': line['revision_id'],
            'text': line['comment'],
        }
        records.append(record)

    with open(output_file, 'w') as f:
        write2json(records, output_file)
    print(len(records))
    # return records

def modify_makeup(wiki_text):
    # replace '<', '>', '&'    
    wiki_text = re.sub("&quot;", "", wiki_text)
    wiki_text = re.sub("&lt;", "<", wiki_text)
    wiki_text = re.sub("&gt;", ">", wiki_text)
    wiki_text = re.sub("&amp;", "&", wiki_text)
    # results = re.compile(r'[http|https]:[a-zA-Z0-9.?/&=:\\\-]*', re.S)
    # wiki_text = re.sub(results, '', wiki_text)
    wiki_text = wiki_text.replace("<ref>", "")
    wiki_text = wiki_text.replace("</ref>", "")
    wiki_text = wiki_text.replace("<\\/ref>", "")
    wiki_text = re.sub(r'\[\[User.*\]\]', '', wiki_text)

    return wiki_text

def filter_data(data, args):
    new_data = []
    for line in data:
        if args.check_comment:
            con = False
            # for word in ["#", "{{", "[[", "template", "image", "infobox", "pic", "link", "photo"]:
            lower_comment = line['comment'].lower()
            for word in ["template", "image", "infobox", "pic", "link", "photo", "comment", "http:", "https:", ".jpg", ".png", 'reply']:
                if word in lower_comment:
                    con = True
                    break
            if con:
                # print(source)
                continue
            for word in ["MOS", "[[WP:CHECKWIKI]]", "[[WP|genfixes]]"]:
                if word in line['comment']:
                    con = True
                    break
            if con:
                # print(source)
                continue
            line['comment'] = line['comment'].replace("using [[Project:AWB|AWB]]", "")
            line['comment'] = line['comment'].replace("using [[Projec|AWB]]", "")
            line['comment'] = line['comment'].replace("using [[Project:AWB]]", "")
            line['comment'] = line['comment'].replace("[[WP|Autowikibrowser]]", "")
            line['comment'] = line['comment'].replace("[[WP|Autowikibrowser run]]", "")
            line['comment'] = line['comment'].replace("with [[WP:AWB|AutoWikiBrowser]]", "")
            line['comment'] = line['comment'].replace("with [[Project:AutoWikiBrowser|AWB]]", "")


            line['comment'] = line['comment'].replace("[[WP:AWB\\/T|typo(s) fixed]]", " typo fixed")
            line['comment'] = line['comment'].replace("[[WP:AWB\\/T|Typo fixing]]", " Typo fixing")
            line['comment'] = line['comment'].replace("[[WP:AWB\\/T|typos fixed]]", " typos fixed")
            line['comment'] = line['comment'].replace("[[WP:AWB\\/T|TypoFix]]", " typos fixed")
            line['comment'] = line['comment'].replace("[[WP:AWB\\/T|typo fixing]]", " typo fixed")
            line['comment'] = line['comment'].replace("[[WP:AWB/T|Typo fixing]]", " Typo fixing")
            line['comment'] = line['comment'].replace("[[WP:AWB/T|typo(s) fixed]]", " typo fixed")
            line['comment'] = line['comment'].replace("[[WP:AWB/T|typos fixed]]", " typos fixed")
            line['comment'] = line['comment'].replace("[[WP:AWB/T|TypoFix]]", " typos fixed")
            line['comment'] = line['comment'].replace("[[WP:AWB/T|typo fixing]]", " typo fixing")
            line['comment'] = line['comment'].replace("[[WP:AWB/T|Typo]]", " typo fixing")
            line['comment'] = line['comment'].replace("[[WP:TSN|Typo fixing]]", " typo fixing")
            line['comment'] = line['comment'].replace("by [[wp:Typo_Team/moss]]", "")
            line['comment'] = line['comment'].replace("by [[Wikipedia:Typo Team/moss]]", "")
            line['comment'] = line['comment'].replace("from [[Wikipedia:Typo Team/moss/R]]", "")


            line['comment'] = line['comment'].replace("[[WP|Typo fixing]]", " typo fixing")
            line['comment'] = line['comment'].replace("[[WP|typos fixed]]", " typos fixed")
            line['comment'] = line['comment'].replace("[[WP|typo fixing]]", " typo fixing")
            line['comment'] = line['comment'].replace("[[WP|typo(s) fixed]]", " typo fixed")
            line['comment'] = line['comment'].replace("[[WP|general]]", " general ")
            line['comment'] = line['comment'].replace("[[WP:RS|reliable sources]]", " reliable sources ")
            line['comment'] = line['comment'].replace("[[WP:RS]]", " reliable sources ")
            line['comment'] = line['comment'].replace("[[WP:OR]]", " original research ")
            line['comment'] = line['comment'].replace("[[WP:EUPHEMISM]]", " euphemisms ")
            line['comment'] = line['comment'].replace("[[WP:EDITORIAL]]", " editorial ")
            line['comment'] = line['comment'].replace("[[WP:TYPO]]", " typo fixing ")
            line['comment'] = line['comment'].replace("[[WP:AWB/T|Typo patrol]]", " typo patrol ")

            line['comment'] = line['comment'].replace("[[WP:EDITORIALISING]]", " editorializing ")


            line['comment'] = line['comment'].replace("[[hyphen]]", "hyphen")
            line['comment'] = line['comment'].replace("[[WP:HYPHEN]]", "hyphen")
            line['comment'] = line['comment'].replace("[[Wikipedia:GENFIXES|general fixes]]", " general fixes")
            line['comment'] = line['comment'].replace("[[WP|general fixes]]", " general fixes")
            line['comment'] = line['comment'].replace("[[WP:AWB/GF|General fixing]]", " general fixing")
            
            line['comment'] = line['comment'].replace("[[WP:NPOV]]", " neutral point of view")
            line['comment'] = line['comment'].replace("[[wp:npov|pov]]", " neutral point of view")
            line['comment'] = line['comment'].replace("[[WP:NPOV|POV]]", " neutral point of view")
            line['comment'] = line['comment'].replace("[[WP:point of view]]", " point of view")
            line['comment'] = line['comment'].replace("[[WP:YESPOV|POV]]", " point of view")
            line['comment'] = line['comment'].replace("[[WP:YESPOV]]", " point of view")

            
            line['comment'] = line['comment'].replace("NPOV", " neutral point of view")
            line['comment'] = line['comment'].replace("npov", " neutral point of view")
            line['comment'] = line['comment'].replace("PoV", " point of view")
            line['comment'] = line['comment'].replace("NPoV", " neutral point of view")
            
            line['comment'] = line['comment'].replace("POV", "point of view")
            line['comment'] = line['comment'].replace(" pov ", " point of view ")
            line['comment'] = line['comment'].replace("pov.", "point of view.")
            line['comment'] = line['comment'].replace("(pov)", "(point of view)")

            line['comment'] = line['comment'].replace("[[WP:AWB|AWB]]", "")

            line['comment'] = line['comment'].replace("with [[Wikipedia:ProveIt|ProveIt]]", "")
            line['comment'] = line['comment'].replace("using [[Wikipedia:ProveIt|ProveIt]]", "")
            line['comment'] = line['comment'].replace("[[Wikipedia|ProveIt]]", " \'prove it\'")

            line['comment'] = line['comment'].replace("[[Wikipedia:ProveIt|ProveIt]]", "")
            # line['comment'] = line['comment'].replace("Cleanup\\/", "Cleanup")
            # line['comment'] = line['comment'].replace("cleanup\\/", "cleanup")
            # line['comment'] = line['comment'].replace("style\\/layout", "style and layout")
            # line['comment'] = line['comment'].replace("Style\\/layout", "Style and layout")
            line['comment'] = line['comment'].replace("Typo/[[WP:AWB/GF|general]]", "Typo ")
            line['comment'] = line['comment'].replace("typo/[[WP:AWB/GF|general]]", "typo")

            line['comment'] = line['comment'].replace("[[WP:AWB/GF|General fixes]]", "General fixes")
            line['comment'] = line['comment'].replace("[[WP:AWB/GF|general fixes]]", "general fixes")
            # line['comment'] = line['comment'].replace("Spelling/grammar", "Spelling and grammar")
            # line['comment'] = line['comment'].replace("spelling/grammar", "spelling and grammar")
            line['comment'] = line['comment'].replace("[[WP:RELTIME]]", "relative time references")
            # line['comment'] = line['comment'].replace("\\/\\/", "")
            line['comment'] = line['comment'].replace("[[WP:AWB|Autowikibrowser run]]", "run")
            line['comment'] = line['comment'].replace("[[WP:AWB|Autowikibrowser run]]", "run")
            line['comment'] = line['comment'].replace("[[WP:PEACOCK|peacock words]]", " point of view and peacock words")
            line['comment'] = line['comment'].replace("[[WP:PEACOCK|peacock word]]", " point of view and peacock word")
            line['comment'] = line['comment'].replace("[[WP:PEACOCK]] term", " point of view and peacock term")
            line['comment'] = line['comment'].replace("[[WP:PEACOCK]]", " point of view and peacock term")
            line['comment'] = line['comment'].replace("[[WP:PEACOCK|peacock term]]", " point of view and peacock term")
            line['comment'] = line['comment'].replace("[[WP:PEACOCK|subjective]]", " subjective and peacock word")
            line['comment'] = line['comment'].replace("[[WP:PEACOCK|peacock]]", " peacock")
            line['comment'] = line['comment'].replace("[[WP:PEACOCK|peacockery]]", " peacock")

            line['comment'] = line['comment'].replace("[[WP:WEASEL|weasel word]]", " weasel word")
            line['comment'] = line['comment'].replace("[[WP:WEASEL|weasel words]]", " weasel words")
            line['comment'] = line['comment'].replace("[[WP:WEASEL]] word", " weasel word")
            line['comment'] = line['comment'].replace("[[WP:WEASEL]]", " weasel word")
            line['comment'] = line['comment'].replace("[[WP:LABEL]]", " contentious labels")
            line['comment'] = line['comment'].replace("[[WP:NOR]]", " no original research")
            line['comment'] = line['comment'].replace("[[WP:OR]]", " original research")
            line['comment'] = line['comment'].replace("[[WP:ORIGINAL RESEARCH]]", " original research")
            line['comment'] = line['comment'].replace("[[WP:ORIGINAL]]", " original research")
            line['comment'] = line['comment'].replace("[[WP:SYN]]", " synthesis")
            line['comment'] = line['comment'].replace("[[WP:SYNTH]]", " synthesis")
            line['comment'] = line['comment'].replace("[[WP:POINT]]", " disrupt Wikipedia to illustrate a point")
            line['comment'] = line['comment'].replace("[[WP:OPED]]", " Opinion Editorial")
            line['comment'] = line['comment'].replace("[[WP:OPED|editorial]]", " Opinion Editorial")



            line['comment'] = modify_makeup(line['comment'])

        if args.check_makeup:
            con = False
            # for word in ["#", "{{", "}}", "[[", "]]", "&"]:
            for word in ["http:", "https:"]:
                if (word in line['src_token']) or (word in line['tgt_token']):
                    con = True
                    break
            if con:
                continue
            line['src_token'] = modify_makeup(line['src_token'])
            line['tgt_token'] = modify_makeup(line['tgt_token'])

        if args.check_lenth:
            if len(line['src_token']) > args.max_lenth or len(line['src_token']) < args.min_lenth:
                continue
            if len(line['tgt_token']) > args.max_lenth or len(line['tgt_token']) < args.min_lenth:
                continue
            if len(line['comment'].split()) < args.comment_min_lenth:
                continue
            if args.max_lenth_difference != -1:
                if len(line['src_token']) - len(line['tgt_token']) > args.max_lenth_difference or len(line['tgt_token']) - len(line['src_token']) > args.max_lenth_difference:
                    continue
            if args.min_lenth_short != -1:
                if len(line['src_token']) - len(line['tgt_token']) < args.min_lenth_short:
                    continue
        if args.check_bleu:
            source = [nltk.word_tokenize(line['src_token'])]
            target = [[nltk.word_tokenize(line['tgt_token'])]]
            bleu_score = corpus_bleu(target, source, smoothing_function=SmoothingFunction().method3)
            if bleu_score >= args.max_bleu:
                # print(source)
                continue
            if bleu_score <= args.min_bleu:
                # print(source)
                continue

        new_data.append(line)
    return new_data

def add_cluster(data, cluster_file, same_order=False):
    cluster = json2list(cluster_file)
    # assert len(data) == len(cluster), f'Please ensure you use the same args when you generate the file for cluster.'
    records = []
    if same_order:
        for d, c in zip(data, cluster):
            record = {
                'id': d['revision_id'],
                'comment': d['comment'],
                'cluster': c['cluster'],
                'src_token': d['src_token'],
                'tgt_token': d['tgt_token'],
            }
            records.append(record)
    else:
        clusters = {}
        for c in cluster:
            clusters[str(c['id'])] = c['cluster']

        for d in data:
            record = {
                'id': d['revision_id'],
                'comment': d['comment'],
                'cluster': clusters[str(d['revision_id'])],
                'src_token': d['src_token'],
                'tgt_token': d['tgt_token'],
            }
            records.append(record)
    return records

def cluster2txt(data, source, target, c):
    s = 0
    with open(source, "a+") as sf, open(target, "a+") as tf:
        for line in data:
            if int(line['cluster']) == c:
                sf.write(line['src_token']+'\n')
                tf.write(line['tgt_token']+'\n')
                s += 1
    print(s)

def merge_outputs(output_path):
    # merge sample results
    print("Merging the sampled outputs from each files ...")
    sample_list = glob.glob(output_path + '*.json')
    sample_file = open(output_path + 'wikicmnt.json', "w", encoding='utf-8')
    for fn in tqdm(sample_list):
        with open(fn, 'r', encoding='utf-8') as fi:
            sample_file.write(fi.read())

def filter_overlap(data, args):
    new_data = []
    delete_data = []
    len_data = len(data)

    for index, line in enumerate(data):
        undigital_comment = re.sub(r'[0-9]+', '', line['comment'])
        undigital_source = re.sub(r'[0-9]+', '', line['src_token'])

        if index < 3:
            start = 0
        else: 
            start = index - 3
        if index > len_data - 4:
            neighbor = data[start:index] + data[index+1:]
        else:
            neighbor = data[start:index] + data[index+1:index+3]
        
        for nei in neighbor:
            # print(nei)
            judgement = corpus_bleu([[nltk.word_tokenize(re.sub(r'[0-9]+', '', nei['comment']))]], [nltk.word_tokenize(undigital_comment)], smoothing_function=SmoothingFunction().method3) > 0.9 and \
                corpus_bleu([[nltk.word_tokenize(re.sub(r'[0-9]+', '', nei['src_token']))]], [nltk.word_tokenize(undigital_source)], smoothing_function=SmoothingFunction().method3) > 0.8
            if judgement:
                break
                
        if judgement:
            delete_data.append(line)
        else:
            new_data.append(line)
    if args.delete_output is not None:
        write2json(delete_data, args.delete_output)
    print('delete ', len(delete_data), ' lines, remain ', len(new_data), ' lines.')
    return new_data

def filter_meaningless_edit(data, args):
    new_data = []
    delete_data = []
    len_data = len(data)

    for index, line in enumerate(data):
        matcher = SequenceMatcher(None, line['src_token'], line['tgt_token'])
        diffs = list(matcher.get_opcodes())
        delete_words = ''
        for diff in diffs:
            tag, i1, i2, j1, j2 = diff
            if tag == 'delete':
                delete_words = delete_words + line['src_token'][i1:i2]
            elif tag == 'replace':
                delete_words = delete_words + line['src_token'][i1:i2]
                
        filter_words = re.sub(r'[0-9\.]+', '', delete_words)
        judgement = len(delete_words) > 0 and len(filter_words.strip()) == 0
        if judgement:
            delete_data.append(line)
        else:
            new_data.append(line)
    if args.delete_output is not None:
        write2json(delete_data, args.delete_output)
    print('delete ', len(delete_data), ' lines, remain ', len(new_data), ' lines.')
    return new_data

def filter_month_edit(data, args):
    new_data = []
    delete_data = []
    len_data = len(data)

    for index, line in enumerate(data):
        source_tokens = nltk.word_tokenize(line['src_token'])
        target_tokens = nltk.word_tokenize(line['tgt_token'])
        matcher = SequenceMatcher(None, source_tokens, target_tokens)
        diffs = list(matcher.get_opcodes())
        delete_words = ''
        for diff in diffs:
            tag, i1, i2, j1, j2 = diff
            if tag == 'replace':
                delete_words = delete_words + ''.join(source_tokens[i1:i2])
        months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        judgement = False
        for m in months:
            if m in delete_words:
                judgement = True
                break

        if judgement:
            delete_data.append(line)
        else:
            new_data.append(line)
    if args.delete_output is not None:
        write2json(delete_data, args.delete_output)
    print('delete ', len(delete_data), ' lines, remain ', len(new_data), ' lines.')
    return new_data

def filter_talk_discuss(data, args):
    new_data = []
    delete_data = []
    len_data = len(data)

    for index, line in enumerate(data):
        judgement = 'talk' in line['comment'].lower() or 'discuss' in line['comment'].lower()

        if judgement:
            delete_data.append(line)
        else:
            new_data.append(line)
    if args.delete_output is not None:
        write2json(delete_data, args.delete_output)
    print('delete ', len(delete_data), ' lines, remain ', len(new_data), ' lines.')
    return new_data

def select_cluster(data, args):
    new_data = []
    for line in data:
        if int(line['cluster']) == args.cluster_num:
            new_data.append(line)
    print(len(new_data), ' is selected in ', len(data))
    return new_data

def main(args):
    if args.func == 'json2txt':
        data = json2list(args.input_file)
        list2txt(data, args.source_file, args.target_file)
    elif args.func == 'json2txt_filter':
        data = json2list(args.input_file)
        data = filter_data(data, args)
        list2txt(data, args.source_file, args.target_file)
    elif args.func == 'json4cluster':
        data = json2list(args.input_file)
        # data = filter_data(data, args)
        list2jsonl(data, args.output_file)
    elif args.func == 'json2cluster':
        data = json2list(args.input_file)
        # data = filter_data(data, args)
        data = add_cluster(data, args.cluster_file)
        write2json(data, args.output_file)
    elif args.func == 'cluster2txt':
        data = json2list(args.input_file)
        for c in range(args.cluster_num):
            source = args.source_file[:-4]+str(c)+args.source_file[-4:]
            target = args.target_file[:-4]+str(c)+args.target_file[-4:]
            cluster2txt(data, source, target, c)
    elif args.func == 'json2txt':
        data = json2list(args.input_file)
        list2txt(data, args.source_file, args.target_file)
    elif args.func == 'merge':
        merge_outputs(args.input_file)
    elif args.func == 'json2json_filter':
        data = json2list(args.input_file)
        data = filter_data(data, args)
        write2json(data, args.output_file)
    elif args.func == 'json2json_filter_overlap':
        data = json2list(args.input_file)
        data = filter_overlap(data, args)
        write2json(data, args.output_file)
    elif args.func == 'json2json_filter_meaningless_edit':
        data = json2list(args.input_file)
        data = filter_meaningless_edit(data, args)
        write2json(data, args.output_file)
    elif args.func == 'json2json_filter_month_edit':
        data = json2list(args.input_file)
        data = filter_month_edit(data, args)
        write2json(data, args.output_file)
    elif args.func == 'json2json_filter_talk_discuss':
        data = json2list(args.input_file)
        data = filter_talk_discuss(data, args)
        write2json(data, args.output_file)
    elif args.func == 'json2json_filter_definate_cluster':
        data = json2list(args.input_file)
        data = select_cluster(data, args)
        data = filter_data(data, args)
        write2json(data, args.output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file',
                        help='Path to the input file',
                        required=True)
    parser.add_argument('-c', '--cluster_file',
                        help='Path to the cluster file',
                        required=False)
    parser.add_argument('-s', '--source_file',
                        help='Path to the source file',
                        required=False)
    parser.add_argument('-t', '--target_file',
                        help='Path to the target file',
                        required=False)
    parser.add_argument('-o', '--output_file',
                        help='Path to the jsonl file',
                        required=False)
    parser.add_argument('--func',
                        type=str,
                        help='function name',
                        default='json2txt')
    parser.add_argument('--max_bleu',
                        type=float,
                        help='',
                        default=100)
    parser.add_argument('--min_bleu',
                        type=float,
                        help='',
                        default=-100)
    parser.add_argument('--check_comment',
                        type=bool,
                        help='',
                        default=True)
    parser.add_argument('--check_makeup',
                        type=bool,
                        help='',
                        default=True)
    parser.add_argument('--check_lenth',
                        type=bool,
                        help='',
                        default=True)
    parser.add_argument('--check_bleu',
                        type=bool,
                        help='',
                        default=True)
    parser.add_argument('--max_lenth',
                        type=int,
                        help='',
                        default=400)
    parser.add_argument('--min_lenth',
                        type=int,
                        help='',
                        default=2)
    parser.add_argument('--max_lenth_difference',
                        type=int,
                        help='',
                        default=-1)
    parser.add_argument('--min_lenth_short',
                        type=int,
                        help='',
                        default=-1)
    parser.add_argument('--cluster_num',
                        type=int,
                        help='',
                        default=8)
    parser.add_argument('--comment_min_lenth',
                        type=int,
                        help='',
                        default=3)
    parser.add_argument('--delete_output',
                        type=str,
                        help='',
                        default=None)
    
    args = parser.parse_args()
    main(args)