import json


def main():
    pretrain_openllm_dataset = []

    # /root/github/upstream-llm-foundry/arc_challenge-0.jsonl
    # {"choices": ["Planetary density will decrease.", "Planetary years will become longer.", "Planetary days will become shorter.", "Planetary gravity will become stronger."], "query": "Question: An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?", "gold": 2}
    arc_challenge = "/root/github/upstream-llm-foundry/arc_challenge-0.jsonl"
    with open(arc_challenge, "r") as f:
        pretrain_dset = []
        for line in f:
            json_object = json.loads(line)
            as_pretrain_sample = json_object['query'] + "\nAnswer: " + json_object['choices'][json_object['gold']]
            pretrain_dset.append(as_pretrain_sample)
        print(f"arc_challenge = {len(pretrain_dset)}")
        pretrain_openllm_dataset.extend(pretrain_dset)

    # /root/github/upstream-llm-foundry/hellaswag-0.jsonl
    # {"query": "Roof shingle removal: A man is sitting on a roof. He", "choices": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "gold": 3}
    hellaswag = "/root/github/upstream-llm-foundry/hellaswag-0.jsonl"
    with open(hellaswag, "r") as f:
        pretrain_dset = []
        for line in f:
            json_object = json.loads(line)
            as_pretrain_sample = json_object['query'] + " " + json_object['choices'][json_object['gold']]
            pretrain_dset.append(as_pretrain_sample)
        print(f"hellaswag = {len(pretrain_dset)}")
        pretrain_openllm_dataset.extend(pretrain_dset)

    # /root/github/upstream-llm-foundry/mmlu-0.jsonl
    # {"query": "Question: For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I slammed on my breaks to miss the ball as it came in to the road. Scenario 2 | I taught my children to play the xylophone.\nChoices\nA. Wrong, Wrong\nB. Wrong, Not wrong\nC. Not wrong, Wrong\nD. Not wrong, Not wrong", "gold": 3, "choices": ["A", "B", "C", "D"], "category": "moral_scenarios"}
    mmlu = "/root/github/upstream-llm-foundry/mmlu-0.jsonl"
    with open(mmlu, "r") as f:
        pretrain_dset = []
        for line in f:
            json_object = json.loads(line)
            as_pretrain_sample = json_object['query'] + "\nAnswer: " + json_object['choices'][json_object['gold']]
            pretrain_dset.append(as_pretrain_sample)
        print(f"mmlu = {len(pretrain_dset)}")
        pretrain_openllm_dataset.extend(pretrain_dset)

    # /root/github/upstream-llm-foundry/winogrande-0.jsonl
    # {"context_options": ["Sarah was a much better surgeon than Maria so Sarah", "Sarah was a much better surgeon than Maria so Maria"], "continuation": "always got the easier cases.", "gold": 1}
    winogrande = "/root/github/upstream-llm-foundry/winogrande-0.jsonl"
    with open(winogrande, "r") as f:
        pretrain_dset = []
        for line in f:
            json_object = json.loads(line)
            as_pretrain_sample = json_object['context_options'][json_object['gold']] + " " + json_object['continuation']
            pretrain_dset.append(as_pretrain_sample)
        print(f"winogrande = {len(pretrain_dset)}")
        pretrain_openllm_dataset.extend(pretrain_dset)

    out_path = "/root/github/upstream-llm-foundry/scripts/data_prep/open_llm/pretrain_openllm.jsonl"
    with open(out_path, 'w') as f:
        for sample in pretrain_openllm_dataset:
            json_line = json.dumps({"text": sample})
            f.write(json_line + "\n")


if __name__ == "__main__":
    main()
