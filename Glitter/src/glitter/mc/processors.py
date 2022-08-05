import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, List, Sequence

from transformers import DataProcessor


@dataclass
class AugmentedMultiChoice:
    id: str
    question: str
    src: Optional[str] = None


@dataclass
class MultiChoiceExample:
    guid: int
    question: str
    choices: Sequence[str]
    label: Optional[int] = None
    augmented_examples: Sequence[AugmentedMultiChoice] = ()

    @property
    def num_choices(self):
        return len(self.choices)

    @property
    def json_dict(self):
        dump = dict(guid=self.guid, question=self.question, choices=self.choices)
        if self.label is not None:
            dump["label"] = self.label
        return dump


@dataclass
class HellaSwagExample(MultiChoiceExample):
    ctx_a: str = None
    ctx_b: str = None
    activity_label: str = None
    split_type: str = None

    @property
    def json_dict(self):
        dump = super().json_dict
        dump["context"] = dump.pop("question")
        dump["ctx_a"] = self.ctx_a
        dump["ctx_b"] = self.ctx_b
        dump["activity_label"] = self.activity_label
        dump["split_type"] = self.split_type
        return dump


class MultiChoiceProcessor(DataProcessor):
    def __init__(self):
        super().__init__()

    def get_example_from_tensor_dict(self, tensor_dict):
        pass

    def get_train_examples(self, data_dir: str):
        data_dir = Path(data_dir)
        train_json_file = data_dir / "train.jsonl"
        return list(self.read_jsonl(str(train_json_file)))

    def get_dev_examples(self, data_dir: str):
        data_dir = Path(data_dir)
        dev_json_file = data_dir / "dev.jsonl"
        return list(self.read_jsonl(str(dev_json_file)))

    def get_test_examples(self, data_dir: str):
        data_dir = Path(data_dir)
        test_json_file = data_dir / "test.jsonl"
        return list(self.read_jsonl(str(test_json_file)))

    def get_labelled_test_examples(self, data_dir: str):
        data_dir = Path(data_dir)
        test_file_path = data_dir / "test.jsonl"
        return list(self.read_jsonl(str(test_file_path)))

    def get_labels(self) -> List[str]:
        raise NotImplementedError()

    @property
    def num_labels(self):
        return len(self.get_labels())

    @classmethod
    def read_jsonl(cls, data_file: str, **kwargs) -> Iterable[MultiChoiceExample]:
        raise NotImplementedError()


class HellaSwagProcessor(MultiChoiceProcessor):
    def get_labels(self) -> List[str]:
        return ["0", "1", "2", "3"]

    @classmethod
    def read_jsonl(cls, data_file: str, **kwargs):
        with open(data_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                ex = json.loads(line.strip())

                augmented_samples = []
                for i, aug_ex in enumerate(ex.get("augmented_samples", [])):
                    augmented_samples.append(
                        AugmentedMultiChoice(
                            f"{id}|{aug_ex.get('id', 'aug-' + str(i + 1))}",
                            aug_ex["question"],
                            aug_ex.get("src", None),
                        )
                    )

                if "choices" not in ex and "endings" not in ex:
                    raise ValueError(f"No endings found for {ex['guid']}")

                yield HellaSwagExample(
                    ex["guid"],
                    ex["context"],
                    ex.get("endings", ex.get("choices", [])),
                    ex["label"],
                    augmented_samples,
                    ex["ctx_a"],
                    ex["ctx_b"],
                    ex["activity_label"],
                    ex["split_type"],
                )


mc_processors = {
    "hellaswag": HellaSwagProcessor,
}