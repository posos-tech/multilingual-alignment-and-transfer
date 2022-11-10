import logging

from multilingual_eval.datasets.chinese_segmenter import StanfordSegmenter


class LabelAlignmentMapper:
    """
    Class for a callable that can be used as argument of datasets.Dataset.map()
    It will perform label alignment for token classification tasks
    """

    def __init__(
        self,
        tokenizer,
        label_name="labels",
        first_subword_only=False,
        max_length=None,
        return_overflowing_tokens=False,
        stride=0,
    ):
        self.tokenizer = tokenizer
        self.label_name = label_name
        self.first_subword_only = first_subword_only
        self.max_length = max_length
        self.return_overflowing_tokens = return_overflowing_tokens
        self.stride = stride

    def __call__(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=self.max_length,
            return_overflowing_tokens=self.return_overflowing_tokens,
            stride=self.stride,
        )

        n = len(tokenized_inputs["input_ids"])

        labels = []
        for i in range(n):
            previous_batch_id = tokenized_inputs.get("overflow_to_sample_mapping", list(range(n)))[
                i
            ]
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            label = examples[self.label_name][previous_batch_id]
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the first_subword_only flag.
                elif self.first_subword_only:
                    label_ids.append(-100)
                else:
                    label_ids.append(label[word_idx])
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs


class StanfordSegmenterWithLabelAlignmentMapper:
    def __init__(self, label_name="labels", relabeling_strategy="first"):
        self.label_name = label_name
        self.relabeling_strategy = relabeling_strategy

        assert relabeling_strategy in ["first", "overlaping_bio", "resegment"]

    def __enter__(self):
        self.segmenter = StanfordSegmenter().__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.segmenter.__exit__(exc_type, exc_val, exc_tb)
        self.segmenter = None

    def __call__(self, example):
        tokens = example["tokens"]
        labels = example[self.label_name]

        if "None" in tokens:
            logging.warning(f"Found 'None' token in sentence: {tokens}")
            tokens = list(map(lambda x: " " if x == "None" else x, tokens))

        if any(map(lambda x: len(x) != 1, tokens)):
            raise Exception(
                f"StanfordSegmenterWithLabelAlignmentMapper expects character-tokenized text. Got: {tokens}"
            )

        sent = "".join(tokens)

        new_tokens = self.segmenter(sent)

        offset = 0

        new_labels = []

        if self.relabeling_strategy == "resegment":
            new_segments = []

        for new_token in new_tokens:
            if self.relabeling_strategy == "first":
                new_labels.append(labels[offset])
            elif self.relabeling_strategy == "overlaping_bio":
                if any(map(lambda x: x > 0, labels[offset : offset + len(new_token)])):
                    new_labels.append(
                        next(
                            iter(
                                filter(
                                    lambda x: x > 0,
                                    labels[offset : offset + len(new_token)],
                                )
                            )
                        )
                    )
                else:
                    new_labels.append(labels[offset])
            elif self.relabeling_strategy == "resegment":
                segments_to_add = []
                labels_to_add = []
                current_segment = ""
                current_label = None
                for char, label in zip(new_token, labels[offset : offset + len(new_token)]):
                    # if current segment was previously empty, create it with current label
                    if current_label is None:
                        current_segment += char
                        current_label = label
                    # if the label is of type B-X, create a new segment
                    elif label % 2 == 1:
                        segments_to_add.append(current_segment)
                        labels_to_add.append(current_label)
                        current_segment = char
                        current_label = label
                    # if the label is equal to previous or previous is B-X and new is I-X, keep the segment together
                    elif current_label == label or (
                        current_label % 2 == 1 and label == current_label + 1
                    ):
                        current_segment += char
                    # otherwise, resegment
                    else:
                        segments_to_add.append(current_segment)
                        labels_to_add.append(current_label)
                        current_segment = char
                        current_label = label

                if len(current_segment) > 0:
                    segments_to_add.append(current_segment)
                    labels_to_add.append(current_label)

                new_segments += segments_to_add
                new_labels += labels_to_add

            offset += len(new_token)

        if self.relabeling_strategy == "resegment":
            new_tokens = new_segments

        return {"tokens": new_tokens, self.label_name: new_labels}

    @classmethod
    def get_language_specific_dataset_transformer(
        cls, label_name="labels", relabeling_strategy="first"
    ):
        def language_specific_transformer(lang, dataset):
            if lang == "zh":
                with cls(label_name, relabeling_strategy) as mapper:
                    return dataset.map(mapper, batched=False)
            return dataset

        return language_specific_transformer
