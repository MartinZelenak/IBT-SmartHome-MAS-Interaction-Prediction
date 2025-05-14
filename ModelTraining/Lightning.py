import argparse
import os
from typing import Any, Dict, List, Type

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.loggers import CSVLogger, Logger, TensorBoardLogger

from Components.DataModules import SequenceDataModule
from Components.LearnerModules import SequenceLearner
from Components.ModelBase import ModelBase
from Components.ModelFactory import ModelFactory
from Components.Models.CfC import PerDeviceCfC
from Components.Models.FC import PerDeviceFC
from Components.Models.LSTM import PerDeviceLSTM
from Components.Models.RNN import PerDeviceRNN

MODEL_DICT: Dict[str, Type[ModelBase]] = {
    "CfC": PerDeviceCfC, 
    "LSTM": PerDeviceLSTM, 
    "RNN": PerDeviceRNN, 
    "FC": PerDeviceFC
}

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as config_file:
        return yaml.safe_load(config_file)

def main():
    # CLI arguments
    model_names = list(MODEL_DICT.keys())
    parser = argparse.ArgumentParser(description="Run an experiment with a selected model.")
    parser.add_argument("-v", "--version", type=str, default=None, help="The version number for this experiment")
    parser.add_argument("-m", "--model", type=str, required=True, choices=model_names, help=f"The name of the model class to use. Available models: {', '.join(model_names)}")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="Path to configuration file")

    ## Config override arguments
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--hidden_size", type=int, help="Override hidden size")
    parser.add_argument("--num_layers", type=int, help="Override number of layers")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--max_epochs", type=int, help="Override max epochs")
    parser.add_argument("--sequence_len", type=int, help="Override sequence length")

    args = parser.parse_args()
    version_name: str | None = args.version
    model_name: str = args.model
    config_path: str = args.config

    config = load_config(config_path)

    if args.batch_size:
        config["datamodule"]["batch_size"] = args.batch_size
    if args.hidden_size:
        config["model"]["hidden_size"] = args.hidden_size
    if args.num_layers:
        config["model"]["num_layers"] = args.num_layers
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.max_epochs:
        config["training"]["max_epochs"] = args.max_epochs
    if args.sequence_len:
        config["datamodule"]["sequence_len"] = args.sequence_len

    dataset_config = config["dataset"]
    datamodule_config = config["datamodule"]
    model_config = config["model"]
    training_config = config["training"]

    model_class = MODEL_DICT.get(model_name)
    if model_class is None:
        raise ValueError(f"Model '{model_name}' is not in the list of available models: {model_names}")

    # Create log folder
    log_folder = config["logging"]["log_folder"]
    os.makedirs(log_folder, exist_ok=True)

    # Data
    time_column_names = dataset_config["time_columns"]
    n_smart_devices = dataset_config["n_smart_devices"]
    n_users = dataset_config["n_users"]
    n_data_columns = len(time_column_names) + n_users + n_smart_devices

    datamodule = SequenceDataModule(
        batch_size=datamodule_config["batch_size"],
        sequence_len=datamodule_config["sequence_len"],
        train_ds_path=dataset_config["train_path"],
        val_ds_path=dataset_config["eval_path"],
        test_ds_path=dataset_config["eval_path"],
        n_data_columns=n_data_columns,
        n_devices=n_smart_devices,
        n_users=n_users,
        num_workers=datamodule_config["num_workers"],
    )

    # Model
    shared_input_size = datamodule.get_feature_size() - n_smart_devices

    model = ModelFactory.create_model(
        model_class,
        shared_input_size=shared_input_size,
        per_device_input_size=1,
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        n_devices=n_smart_devices,
    )

    # Experiment
    keep_hidden_state = training_config["keep_hidden_state"]
    sequence_len = datamodule_config["sequence_len"]

    ## Model either gets the sequence or remembers it
    assert (
        not keep_hidden_state or sequence_len == 1
    ), f"Got keep_hidden_state <{'true' if keep_hidden_state else 'false'}> and sequence_length <{sequence_len}>. Model either gets the whole sequence or remembers the information in hidden_state!"

    learner = SequenceLearner(
        model,
        training_config["learning_rate"],
        keep_hidden_state=keep_hidden_state,
        n_devices=n_smart_devices,
        sequence_len=sequence_len,
    )

    tb_logger = TensorBoardLogger(f"{log_folder}/tb", name=model_name, version=version_name)
    csv_logger = CSVLogger(f"{log_folder}/csv", name=model_name, version=version_name)
    loggers: List[Logger] = [
        tb_logger,
        csv_logger
    ]
    print(f"TB  output folder: {tb_logger.log_dir}")
    print(f"CSV output folder: {csv_logger.log_dir}")

    profiler = None
    # profiler = PyTorchProfiler(
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_logger.log_dir), # type: ignore
    #     schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20)
    # )

    torch.set_float32_matmul_precision("high")

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=loggers,
        profiler=profiler,
        min_epochs=1,
        max_epochs=training_config["max_epochs"],
        precision="32",
        log_every_n_steps=1,
    )

    trainer.fit(learner, datamodule=datamodule)

    learner.add_custom_scalers()

if __name__ == "__main__":
    main()
