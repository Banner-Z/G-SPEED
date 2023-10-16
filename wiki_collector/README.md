# Wikipedia Editing History Collector

We collect Wikipedia editing history following [WikiCommentEdit](https://github.com/microsoft/WikiCommentEdit.git).
It is worth noting that we have made some changes in implementation details, such as: we retain the complete sentence instead of intercepting the three words before and after the modified content and we use [wikiextractor](https://github.com/attardi/wikiextractor) to remove markup (you need to download it). This is why we provide this code.

You can refer to the description [here](https://github.com/microsoft/WikiCommentEdit/tree/master/data_processing#download-wiki-dump-files) to get a `dumpstatus.json` file.
Then:
```
cd data_processing
bash process.sh
```

After collection, the data needs to be filtered. Some references are given in `prepare.py`.
