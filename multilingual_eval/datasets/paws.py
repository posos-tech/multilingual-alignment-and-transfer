class PAWSMapper:
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        res = self.tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            max_length=self.max_length,
            truncation=True,
        )
        return {**res, "label": examples["label"]}
