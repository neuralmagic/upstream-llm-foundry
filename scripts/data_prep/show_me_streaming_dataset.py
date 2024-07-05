from streaming.base import StreamingDataset
from transformers import AutoTokenizer

mdl = 'meta-llama/Meta-Llama-3-8B'
dset = '/nm/drive0/eldar/datasets/llama3_8b_Cosmopedia_seq8k_tokenized_wEOS_noWrap'
split = 'train'
how_many = "10"

tokenizer = AutoTokenizer.from_pretrained(mdl)
ds = StreamingDataset(local=dset, split=split)

for i, item in enumerate(ds):
    if how_many == "all":
        print(f"\n--- {i}/{len(ds)} ---")
    else:
        print(f"\n--- {i}/{how_many} ---")

    # print(f"PROMPT = {tokenizer.decode(item['turns'][0]['input_ids'])}")
    # print(f"RESPONSE = {tokenizer.decode(item['turns'][0]['labels'])}")

    len1 = len(item['turns'][0]['input_ids'])
    len2 = len(item['turns'][0]['labels'])
    assert len1 + len2 <= 1024, f"Problem with {len1} and {len2}"
    # print(f"PROMPT = {tokenizer.decode(item['turns'][0]['input_ids'])}")
    # print(f"RESPONSE = {tokenizer.decode(item['turns'][0]['labels'])}")

    if how_many != "all":
        how_many -= 1
        if how_many == 0:
            break

# for i in range(how_many):
#     # note, you need to copy the numpy array because the original is non-writeable
#     # and torch does not support non-writeable tensors, so you get a scary warning and
#     # if you do try to write to the tensor you get undefined behavior
#     print(f"PROMPT = {tokenizer.decode(ds[i]['turns'][0]['input_ids'])}")
#     print(f"RESPONSE = {tokenizer.decode(ds[i]['turns'][0]['labels'])}")


        # import torch
        # from streaming.base import StreamingDataset
        # from transformers import AutoTokenizer

        # tokenizer = AutoTokenizer.from_pretrained('your/tokenizer')
        # ds = StreamingDataset(local='mds-data-folder', split='val')

        # # note, you need to copy the numpy array because the original is non-writeable
        # # and torch does not support non-writeable tensors, so you get a scary warning and
        # # if you do try to write to the tensor you get undefined behavior
        # tokens = torch.from_numpy(np.frombuffer(ds[0]['tokens'], dtype=np.int64).copy())
<<<<<<< HEAD
        # print(tokenizer.decode(tokens))
=======
        # print(tokenizer.decode(tokens))
>>>>>>> 4ce3d69 (Add support for more datasets and tokenizer_call_kwargs)
