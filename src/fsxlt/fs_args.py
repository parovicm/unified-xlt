from dataclasses import dataclass, field
from typing import Optional

@dataclass
class FewShotArguments:
    multisource_data: Optional[str] = field(
        default=None, metadata={"help": "JSON multi-source dataset descriptor"}
    )
    language_adaptation: bool = field(
        default=False, metadata={"help": "Whether to do language adaptation during multi-source training"}
    )
    source_lang: Optional[str] = field(
        default=None, metadata={"help": "The name of the source language for cross-lingual transfer."}
    )
    target_lang: Optional[str] = field(
        default=None, metadata={"help": "The name of the target language for cross-lingual transfer."}
    )

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use for training and evaluation."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset subset to use for training and evaluation."}
    )

    source_dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the source dataset to use if different from the target dataset."}
    )
    source_has_config: bool = field(
        default=True,
        metadata={
            "help": "Whether to select a subset of source dataset with --source_config_name."
        }
    )
    source_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the dataset config for the source language "
                    "(only needed if it's not the same as --source_lang)."
        }
    )
    source_train_file: Optional[str] = field(
        default=None, metadata={"help": "The name of the file containing training data for the source language."}
    )
    source_validation_file: Optional[str] = field(
        default=None, metadata={"help": "The name of the file containing validation data for the source language."}
    )
    source_test_file: Optional[str] = field(
        default=None, metadata={"help": "The name of the file containing test data for the source language."}
    )
    source_multisource_data: Optional[str] = field(
        default=None, metadata={
            "help": "Source language multisource dataset JSON, "
                    "to be used if source data is taken from more than one dataset."
        }
    )

    target_dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the target dataset to use if different from the source dataset."}
    )
    target_has_config: bool = field(
        default=True,
        metadata={
            "help": "Whether to select a subset of target dataset with --target_config_name."
        }
    )
    target_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the dataset config for the target language "
                    "(only needed if it's not the same as --target_lang)."
        }
    )
    target_train_file: Optional[str] = field(
        default=None, metadata={"help": "The name of the file containing training data for the target language."}
    )
    target_validation_file: Optional[str] = field(
        default=None, metadata={"help": "The name of the file containing validation data for the target language."}
    )
    target_test_file: Optional[str] = field(
        default=None, metadata={"help": "The name of the file containing test data for the target language."}
    )
    target_train_split: Optional[str] = field(
        default=None, metadata={"help": "The name of the target split to use as training split."}
    )
    target_multisource_data: Optional[str] = field(
        default=None, metadata={
            "help": "Target language multisource dataset JSON, "
                    "to be used if target data is taken from more than one dataset."
        }
    )

    adaptation_method: Optional[str] = field(
        default='sft', metadata={"help": '"sft" for sparse fine-tuning, "adapters" for adapters.'}
    )
    source_adaptation: bool = field(
        default=False, metadata={"help": "Whether to apply language adaptation for the source language."}
    )
    target_adaptation: bool = field(
        default=False, metadata={"help": "Whether to apply language adaptation for the target language."}
    )
    lang_adapt_format: Optional[str] = field(
        default=None, metadata={"help": "Pattern where replacing {} with the language name yields its language adapter/SFT path."}
    )

    source_num_examples: Optional[int] = field(
        default=-1, metadata={"help": "Max source train examples. If -1, all available will be used."}
    )
    target_num_examples: Optional[int] = field(
        default=-1, metadata={"help": "Max target train examples. If -1, all available will be used."}
    )
    target_upsampling: Optional[int] = field(
        default=1, metadata={"help": "Factor by which to upsample target language examples during training."}
    )
    include_target_in_evaluation: bool = field(
        default=True, metadata={"help": "Whether to include target language data in evaluation set."}
    )
    include_source_in_evaluation: bool = field(
        default=True, metadata={"help": "Whether to include source language data in the evaluation."}
    )

    def __post_init__(self):
        if self.dataset_name is not None:
            self.source_dataset_name = self.dataset_name
            self.target_dataset_name = self.dataset_name

        if self.source_config_name is None and self.source_has_config:
            self.source_config_name = self.source_lang
        if self.target_config_name is None and self.target_has_config:
            self.target_config_name = self.target_lang

