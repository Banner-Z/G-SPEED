import argparse
import gzip
import itertools
import json
import numpy as np
import pandas as pd
import pickle
import torch
import tqdm
from itertools import chain
from pathlib import Path
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
from typing import Dict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from numba import config
config.CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY = True
import cuml

class ShardedIterator(IterableDataset):
    """A sharded wrapper around an iterable, padded to length.

    Args:
        iterable (iterable): iterable to wrap
        num_shards (int): number of shards to split the iterable into
        shard_id (int): which shard to iterator over
        fill_value (Any, optional): padding value when the iterable doesn't
            evenly divide *num_shards* (default: None).
    """

    def __init__(self, iterable, num_shards, shard_id, fill_value=None):
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError('shard_id must be between 0 and num_shards')

        self._sharded_len = len(iterable) // num_shards
        if len(iterable) % num_shards > 0:
            self._sharded_len += 1

        self.itr = itertools.zip_longest(
            range(self._sharded_len),
            itertools.islice(iterable, shard_id, len(iterable), num_shards),
            fillvalue=fill_value,
        )

    def __len__(self):
        return self._sharded_len

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.itr)[1]

        
class IterableDomain(IterableDataset):
    def __init__(self,
                domain_directory: Path=None,
                file: Path=None,
                add_bos_token: bool = False,
                bos_token: str = "<|endoftext|>",
                sample_files: int = None,
                track_token_count: bool = False,
                silent: bool=False,
                ignore_ids: Dict[str, int]={},
                text_field: str = "text",
                json: bool = True,
                path_to_clusters: Path=None
                ):
        if file is not None:
            self.files = file
            self.num_files = 1
        else:
            self.files = list(chain(domain_directory.glob(r"*.json.gz"),
                                    domain_directory.glob(r"*.jsonl.gz"),
                                    domain_directory.glob(r"*.txt.gz")))
            if sample_files:
                self.files = np.random.choice(self.files, sample_files)
            self.num_files = len(self.files)
        self.add_bos_token = add_bos_token
        self.track_token_count = track_token_count
        self.bos_token = bos_token
        self.domain_directory = domain_directory
        self.text_field = text_field
        self.ignore_ids = ignore_ids
        self.json = json
        self.path_to_clusters = path_to_clusters
        self.num_docs = 0
        self.count_lines()

    def count_lines(self):
        if torch.utils.data.get_worker_info() is not None:
            worker_total_num = torch.utils.data.get_worker_info().num_workers
            worker_id = torch.utils.data.get_worker_info().id
        else:
            worker_total_num = 1
            worker_id = 0
        for json_file in tqdm(itertools.islice(self.files,
                                               worker_id,
                                               None,
                                               worker_total_num),
                              total=self.num_files):
            if str(json_file).endswith(".gz"):
                file_itr = gzip.open(json_file, 'rb')
            else:
                file_itr = open(json_file, 'r')
            for ix in tqdm(file_itr):
                self.num_docs += 1

    def line_mapper(self, line):
        if self.json:
            sample = json.loads(line)
            try:
                sample.get(self.text_field)
            except:
                sample = {self.text_field : line}
        else:
            sample = {self.text_field: line}
        if not sample.get(self.text_field):
            return None, None, None
        token_count = len(sample[self.text_field].split())
        text = sample.pop(self.text_field)
        if self.add_bos_token:
            text = self.bos_token + " " + text
        return text, token_count, sample
    
    def __len__(self):
        return self.num_docs

    def __iter__(self):
        if torch.utils.data.get_worker_info() is not None:
            worker_total_num = torch.utils.data.get_worker_info().num_workers
            worker_id = torch.utils.data.get_worker_info().id
        else:
            worker_total_num = 1
            worker_id = 0
        if self.path_to_clusters:
            cluster_itr = np.load(self.path_to_clusters)
        for json_file in tqdm(itertools.islice(self.files,
                                                worker_id,
                                                None,
                                                worker_total_num),
                                disable=True,
                                total=self.num_files):
            if str(json_file).endswith(".gz"):
                file_itr = gzip.open(json_file, 'rb')
            else:
                file_itr = open(json_file, 'r')
            mapped_itr = map(self.line_mapper, file_itr)
            
            for ix, item in enumerate(mapped_itr):
                if item[0] is not None:
                    res = {'id': ix, 'file': str(json_file), 'text': item[0], 'token_count': item[1]}
                    if self.path_to_clusters:
                        res['cluster'] = cluster_itr[ix]
                    yield res


def load_model(path_to_model: Path):
    with open(path_to_model, 'rb') as f:
        out = pickle.load(f)
    return out


def get_texts(dataset):
    dataset = IterableDomain(file=dataset)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=512)
    domains = []
    for batch in tqdm(dataloader):
        for item in zip(batch['id'], batch['text']):
            domains.append({"id": item[0].item(), "text": item[1]})
    return pd.DataFrame(domains)


def transformer(vectorizer, dataset, num_parallel_shards, i):
    vecs = []
    iterator_ = ShardedIterator(dataset, num_parallel_shards, shard_id=i, fill_value = {})
    vecs = vectorizer.transform([x['text'] for x in iterator_ if x])
    return vecs


def vectorize_sklearn(model, file, batch_size=512, parallel=False, num_parallel_shards=10):
    dataset = IterableDomain(file=file)
    if parallel:
        import joblib
        from joblib import delayed
        vecs = joblib.Parallel(n_jobs=num_parallel_shards)(delayed(transformer)(model, 
                                                                                dataset,
                                                                                num_parallel_shards,
                                                                                i) 
                                                            for i  in tqdm(range(num_parallel_shards)))
        vecs = np.concatenate(vecs, 0)
    else:
        vecs = vectorizer.transform(tqdm([x['text'] for x in dataset]))
    return vecs


def get_files(data_dir):
    directory = Path(data_dir)
    files = list(directory.glob(r"*.jsonl"))
    print(directory)
    print(files)
    return files


def get_top_terms(vectorizer, kmeans):
    # this will only work if you use TFIDF vectorizer (which maintains vocab)
    original_space_centroids = vectorizer['svd'].inverse_transform(kmeans.cluster_centers.cpu())
    order_centroids = original_space_centroids.argsort()[:, ::-1]
    vocab = vectorizer['tfidf'].get_feature_names_out()
    top_terms = []
    for i in range(kmeans.n_clusters):
        terms = {}
        for j in range(10):
            terms[f'term_{j}'] = vocab[order_centroids[i, j]]
        top_terms.append(terms)
    return pd.DataFrame(top_terms)


def number_normalizer(tokens):
    """Map all numeric tokens to a placeholder.

    For many applications, tokens that begin with a number are not directly
    useful, but the fact that such a token exists can be relevant.  By applying
    this form of dimensionality reduction, some methods may perform better.
    """
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
    # this vectorizer replaces numbers with #NUMBER token
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))


def train_vectorizer(file, model_name, path_to_vectorizer, do_svd=False, norm=False):
    texts = get_texts(file)
    model = SentenceTransformer(model_name)
    # print(list(texts.text[:]))
    embeddings = model.encode(list(texts.text[:]))
    print(embeddings.shape)
    vectorizer = None
    if do_svd and norm:
        vectorizer = Pipeline([('svd', TruncatedSVD(n_components=100)),
                        ('normalizer', Normalizer(copy=False))])
    elif do_svd:
        vectorizer = Pipeline([('svd', TruncatedSVD(n_components=100)),
                        ])
    else:
        return embeddings
    vecs = vectorizer.fit_transform(embeddings)
    with open(path_to_vectorizer,  'wb+') as f:
        _ = pickle.dump(vectorizer, f)
    
    return vectorizer, vecs

def get_vector(vectorizer, text, model_name):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(list(text))
    if vectorizer is not None:
        embeddings = vectorizer.transform(embeddings)
    print(embeddings.shape)
    return embeddings

def train_kmeans(vecs, path_to_kmeans, args):
    # if torch.cuda.is_available():
    #     device = torch.device('cuda')
    # else:
    #     device = torch.device('cpu')
    kmeans_float  = cuml.KMeans(n_clusters=args.num_clusters, init=args.init_type, oversampling_factor=args.oversampling_factor, \
                                max_iter=args.max_iter, n_init=args.n_init)
    y_pred = kmeans_float .fit_predict(vecs)
    # y_pred = DBSCAN(eps = eps, min_samples = min_samples).fit_predict(vecs)
    with open(path_to_kmeans,  'wb+') as f:
        _ = pickle.dump(kmeans_float, f)
    return y_pred


def main(args,
         data_dir,
         output_dir=Path("cluster_output/"),
         model_name='sentence-transformers/all-mpnet-base-v2'):
    files = get_files(data_dir)
    metadata = []
    for file in tqdm(files):
        metadata.append(pd.read_json(file, lines=True))
    metadata = pd.concat(metadata, axis=0)
    # print(metadata)
    # raise 'end'
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.path_to_vec is None and args.path_to_vectorizer is None:
        path_to_vec = output_dir / "vecs.pkl"
        path_to_vectorizer = output_dir / "svd.pkl"
        vectorizer, vecs = train_vectorizer(files, model_name, path_to_vectorizer, do_svd=args.do_svd, norm=args.norm)
        with open(path_to_vec,  'wb+') as f:
            _ = pickle.dump(vecs, f)
    else:
        with open(args.path_to_vec, 'rb') as f:
            vecs = pickle.load(f)
        with open(args.path_to_vectorizer, 'rb') as f:
            vectorizer = pickle.load(f)
    path_to_kmeans = output_dir / "kmeanspp.pkl"
    y_pred = train_kmeans(vecs, path_to_kmeans, args)
    metadata['cluster'] = y_pred

    return vectorizer, metadata


def evaluate(vectorizer, kmeans, data_dir, model_name):
    files = get_files(data_dir)
    metadata = []
    for file in tqdm(files):
        metadata.append(pd.read_json(file, lines=True))
    metadata = pd.concat(metadata, axis=0)
    # metadata = metadata.sample(min(20000, metadata.shape[0]))
    datas = None
    for i in range(len(metadata['id'])):
        start = i*10000
        end = (i+1)*10000
        if end>=len(metadata['id']):
            end = len(metadata['id'])-1
            texts = metadata.text[start:]
        else:
            texts = metadata.text[start:end]
        # print(get_vector(vectorizer, texts, model_name))
        data = kmeans.predict(torch.from_numpy(get_vector(vectorizer, texts, model_name)))
        if datas != None:
            datas = torch.cat((datas,data), 0)
        else:
            datas = data
        if end>=len(metadata['id'])-1:
            break
    metadata['cluster'] = datas
    return metadata

def write2json(records, output_dir):
    with open(output_dir, "w", encoding="utf8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

if __name__ == '__main__':

    parser = argparse.ArgumentParser() 
    parser.add_argument('--data-dir', required=True, type=str)
    parser.add_argument('--num-clusters', required=True, type=int)
    parser.add_argument('--balanced', action='store_true')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--do_svd', default=False, type=bool)
    parser.add_argument('--norm', default=False, type=bool)
    parser.add_argument('--init_type', default='scalable-k-means++', type=str)
    parser.add_argument('--oversampling_factor', default=2, type=int)
    parser.add_argument('--n_init', default=1, type=int)
    parser.add_argument('--max_iter', default=300, type=int)
    parser.add_argument('--path_to_vec', default=None, type=str)
    parser.add_argument('--path_to_vectorizer', default=None, type=str)


    args = parser.parse_args()
    model_name = 'sentence-transformers/all-mpnet-base-v2'

    if not args.eval_only:
        vectorizer, metadata = main(args,
                                data_dir=args.data_dir,
                                output_dir=args.output_dir,
                                model_name=model_name)
        
    meta_dir = args.output_dir / "meta_data_pred.json"
    # print(metadata)
    records = []
    # print(metadata)
    for i, t, c in zip(metadata.id, metadata.text, metadata.cluster):
        record = {
            'id': i,
            'text': t,
            'cluster': c,
        }
        records.append(record)
    # print(records)
    write2json(records, meta_dir)