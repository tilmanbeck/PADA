import os

from src.utils.constants import PROJECT_ROOT_DIR, DATA_DIR, EXP_DIR
from src.data_processing.absa.pada import AbsaSeq2SeqPadaDataProcessor, AbsaSeq2SeqPadaDataset
from src.data_processing.rumor.pada import RumorPadaDataProcessor, RumorPadaDataset
from src.data_processing.stab2018.pada import Stab2018PadaDataProcessor, Stab2018PadaDataset, stab2018_domain_label_mapping
from src.data_processing.stab2018.base import Stab2018DataProcessor, Stab2018Dataset
from src.modeling.token_classification.pada_seq2seq_token_classifier import PadaSeq2SeqTokenClassifierGeneratorMulti
from src.modeling.text_classification.pada_text_classifier import PadaTextClassifierMulti
from src.modeling.text_classification.t5_text_classifier import T5TextClassifier
from src.utils.train_utils import set_seed, LoggingCallback
from pathlib import Path
from argparse import Namespace, ArgumentParser
from pytorch_lightning import Trainer
from syct import timer
os.environ['TOKENIZERS_PARALLELISM']="false"

SUPPORTED_MODELS = {
    "PADA-rumor": (PadaTextClassifierMulti, RumorPadaDataProcessor, RumorPadaDataset),
    "PADA-absa": (PadaSeq2SeqTokenClassifierGeneratorMulti, AbsaSeq2SeqPadaDataProcessor, AbsaSeq2SeqPadaDataset),
    "PADA-stab2018": (PadaTextClassifierMulti, Stab2018PadaDataProcessor, Stab2018PadaDataset),
    "T5-stab2018": (T5TextClassifier, Stab2018DataProcessor, Stab2018Dataset)
}

SUPPORTED_DATASETS = {
    "rumor",
    "absa",
    "stab2018"
}

DOMAIN_LABEL_MAPPINGS = {
    'stab2018': stab2018_domain_label_mapping
}

args_dict = dict(
    ckpt_path=str(PROJECT_ROOT_DIR / "checkpoints"),
    model_name="PADA",
    dataset_name="rumor",
    # dataset_name="absa",
    src_domains="sydneysiege,ferguson,germanwings-crash,ottawashooting",
    # src_domains="laptops,rest,service",
    trg_domain="charliehebdo",
    # trg_domain="device",
    data_dir=str(DATA_DIR),  # path to data files
    experiment_dir=str(EXP_DIR),  # path to base experiment dir
    output_dir=str(EXP_DIR),  # path to save the checkpoints
    eval_batch_size=32,
    n_gpu=1,
    seed=41,
    beam_size=10,
    repetition_penalty=2.0,
    length_penalty=1.0,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True,
    num_return_sequences=4,
    num_beam_groups=5,
    diversity_penalty=0.2,
    #    eval_metrics=["binary_f1", "micro_f1", "macro_f1", "weighted_f1"],
    eval_metrics=["micro_f1", "macro_f1", "weighted_f1"],
    proportion_aspect=0.3333,
    gen_constant=1.0,
    multi_diversity_penalty=1.0,
    replace_domain_label=False
)


@timer
def eval_trained_pada_model(args):
    if isinstance(args, Namespace):
        hparams = args
    elif isinstance(args, dict):
        hparams = Namespace(**args)

    hparams.src_domains = hparams.src_domains.split(",")

    experiment_name = f"{hparams.dataset_name.lower()}_{hparams.trg_domain}_{hparams.model_name}_eval-ckpt"
    hparams.output_dir = Path(hparams.output_dir) / experiment_name.replace("_", "/")
    hparams.output_dir.mkdir(exist_ok=True, parents=True)
    hparams.output_dir = str(hparams.output_dir)

    logging_callback = LoggingCallback()
    logger = True
    callbacks = [logging_callback]
    ckpt_path = Path(hparams.ckpt_path)
    if ckpt_path.is_file() and ckpt_path.suffix == ".ckpt":
        test_ckpt = hparams.ckpt_path
        experiment_dir = hparams.experiment_dir
    elif ckpt_path.is_dir():
        test_ckpt = f"{hparams.ckpt_path}/{hparams.dataset_name}/{hparams.trg_domain}/PADA/best_dev_binary_f1.ckpt"
        experiment_dir = hparams.ckpt_path
    else:
        raise ValueError("Error - ckpt_path parameter should point to a directory or a .ckpt file!")

    model_hparams_dict = vars(hparams)

    train_args = dict(
            default_root_dir=model_hparams_dict["output_dir"],
            gpus=model_hparams_dict["n_gpu"],
            callbacks=callbacks,
            logger=logger,
            deterministic=True,
            benchmark=False
        )

    set_seed(model_hparams_dict.pop("seed"))
    dataset_name = model_hparams_dict.pop("dataset_name")
    if dataset_name in ["rumor", "mnli", "stab2018"]:
        model_hparams_dict.pop("proportion_aspect")
        model_hparams_dict.pop("multi_diversity_penalty")
    else:
        model_hparams_dict.pop("gen_constant")
    model_name = model_hparams_dict.pop("model_name")

    if model_name in ["T5"]:
        for param in ["proportion_aspect", "multi_diversity_penalty", "max_drf_seq_len", "gen_constant", "mixture_alpha", "num_return_sequences"]:
            if param in model_hparams_dict:
                model_hparams_dict.pop(param)

    replace_domain_label = args.replace_domain_label

    model_obj, data_procesor_obj, dataset_obj = SUPPORTED_MODELS[f"{model_name}-{dataset_name}"]
    model = model_obj.load_from_checkpoint(checkpoint_path=test_ckpt,
                                           eval_batch_size=hparams.eval_batch_size,
                                           data_dir=hparams.data_dir,
                                           experiment_dir=experiment_dir,
                                           output_dir=hparams.output_dir).eval()
    # indicate if generated domain labels should be replaced by target domain label
    model.replace_domain_label = replace_domain_label
    # pass label mapping
    model.domain_label_mapping = DOMAIN_LABEL_MAPPINGS[dataset_name]
    trainer = Trainer(**train_args)
    trainer.test(model)


def main():
    parser = ArgumentParser()
    for key, val in args_dict.items():
        if key == "dataset_name":
            parser.add_argument(f"--{key}", default=val, type=type(val),
                                choices=SUPPORTED_DATASETS)
        elif key == "model_name":
            parser.add_argument(f"--{key}", default=val, type=type(val),
                                choices=("PADA","T5"))
        elif type(val) is bool:
            parser.add_argument(f"--{key}", default=val, action="store_true", required=False)
        else:
            parser.add_argument(f"--{key}", default=val, type=type(val), required=False)
    args = parser.parse_args()
    eval_trained_pada_model(args)


if __name__ == "__main__":
    main()
