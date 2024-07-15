from datasets.builder import *
from datasets.load import *
from datasets.packaged_modules import _EXTENSION_TO_MODULE, _PACKAGED_DATASETS_MODULES
from datasets.naming import camelcase_to_snakecase
import os


def get_cache_dir(dataset_name):
    namespace_and_dataset_name = dataset_name.split("/")
    namespace_and_dataset_name[-1] = camelcase_to_snakecase(namespace_and_dataset_name[-1])
    cached_relative_path = "___".join(namespace_and_dataset_name)
    return cached_relative_path

class MyDatasetBuilder(DatasetBuilder):

    def as_dataset(
            self,
            split: Optional[Split] = None,
            run_post_process=True,
            verification_mode: Optional[Union[VerificationMode, str]] = None,
            ignore_verifications="deprecated",
            in_memory=False,
    ) -> Union[Dataset, DatasetDict]:
        if ignore_verifications != "deprecated":
            verification_mode = verification_mode.NO_CHECKS if ignore_verifications else VerificationMode.ALL_CHECKS
            warnings.warn(
                "'ignore_verifications' was deprecated in favor of 'verification' in version 2.9.1 and will be removed in 3.0.0.\n"
                f"You can remove this warning by passing 'verification_mode={verification_mode.value}' instead.",
                FutureWarning,
            )
        if self._file_format is not None and self._file_format != "arrow":
            raise FileFormatError('Loading a dataset not written in the "arrow" format is not supported.')
        is_local = not is_remote_filesystem(self._fs)
        if not is_local:
            raise NotImplementedError(f"Loading a dataset cached in a {type(self._fs).__name__} is not supported.")
        if not os.path.exists(self._output_dir):
            raise FileNotFoundError(
                f"Dataset {self.dataset_name}: could not find data in {self._output_dir}. Please make sure to call "
                "builder.download_and_prepare(), or use "
                "datasets.load_dataset() before trying to access the Dataset object."
            )

        logger.debug(f'Constructing Dataset for split {split or ", ".join(self.info.splits)}, from {self._output_dir}')

        # By default, return all splits
        if split is None:
            split = {s: s for s in self.info.splits}

        verification_mode = VerificationMode(verification_mode or VerificationMode.BASIC_CHECKS)

        # Create a dataset for each of the given splits
        datasets = map_nested(
            partial(
                self._build_single_dataset,
                run_post_process=run_post_process,
                verification_mode=verification_mode,
                in_memory=in_memory,
            ),
            split,
            map_tuple=True,
            disable_tqdm=False, #change here
        )
        if isinstance(datasets, dict):
            datasets = DatasetDict(datasets)
        return datasets


def my_load_dataset_builder(
    path: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    cache_dir: Optional[str] = None,
    features: Optional[Features] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[Union[DownloadMode, str]] = None,
    revision: Optional[Union[str, Version]] = None,
    token: Optional[Union[bool, str]] = None,
    use_auth_token="deprecated",
    storage_options: Optional[Dict] = None,
    **config_kwargs,
) -> MyDatasetBuilder:
    if use_auth_token != "deprecated":
        warnings.warn(
            "'use_auth_token' was deprecated in favor of 'token' in version 2.14.0 and will be removed in 3.0.0.\n"
            f"You can remove this warning by passing 'token={use_auth_token}' instead.",
            FutureWarning,
        )
        token = use_auth_token
    download_mode = DownloadMode(download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS)
    if token is not None:
        download_config = download_config.copy() if download_config else DownloadConfig()
        download_config.token = token
    if storage_options is not None:
        download_config = download_config.copy() if download_config else DownloadConfig()
        download_config.storage_options.update(storage_options)
    dataset_module = dataset_module_factory(
        path,
        revision=revision,
        download_config=download_config,
        download_mode=download_mode,
        data_dir=data_dir,
        data_files=data_files,
    )
    # Get dataset builder class from the processing script
    builder_kwargs = dataset_module.builder_kwargs
    data_dir = builder_kwargs.pop("data_dir", data_dir)
    data_files = builder_kwargs.pop("data_files", data_files)
    config_name = builder_kwargs.pop(
        "config_name", name or dataset_module.builder_configs_parameters.default_config_name
    )
    dataset_name = builder_kwargs.pop("dataset_name", None)
    hash = builder_kwargs.pop("hash")
    info = dataset_module.dataset_infos.get(config_name) if dataset_module.dataset_infos else None
    if (
        dataset_module.builder_configs_parameters.metadata_configs
        and config_name in dataset_module.builder_configs_parameters.metadata_configs
    ):
        hash = update_hash_with_config_parameters(
            hash, dataset_module.builder_configs_parameters.metadata_configs[config_name]
        )

    if path in _PACKAGED_DATASETS_MODULES and data_files is None:
        error_msg = f"Please specify the data files or data directory to load for the {path} dataset builder."
        example_extensions = [
            extension for extension in _EXTENSION_TO_MODULE if _EXTENSION_TO_MODULE[extension] == path
        ]
        if example_extensions:
            error_msg += f'\nFor example `data_files={{"train": "path/to/data/train/*.{example_extensions[0]}"}}`'
        raise ValueError(error_msg)

    builder_cls = get_dataset_builder_class(dataset_module, dataset_name=dataset_name)
    # Instantiate the dataset builder
    builder_instance= MyDatasetBuilder(
        cache_dir=cache_dir,
        dataset_name=dataset_name,
        config_name=config_name,
        data_dir=data_dir,
        data_files=data_files,
        hash=hash,
        info=info,
        features=features,
        token=token,
        storage_options=storage_options,
        **builder_kwargs,
        **config_kwargs,
    )

    return builder_instance

def my_load_dataset(
    path: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    split: Optional[Union[str, Split]] = None,
    cache_dir: Optional[str] = None,
    features: Optional[Features] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[Union[DownloadMode, str]] = None,
    verification_mode: Optional[Union[VerificationMode, str]] = None,
    ignore_verifications="deprecated",
    keep_in_memory: Optional[bool] = None,
    save_infos: bool = False,
    revision: Optional[Union[str, Version]] = None,
    token: Optional[Union[bool, str]] = None,
    use_auth_token="deprecated",
    task="deprecated",
    streaming: bool = False,
    num_proc: Optional[int] = None,
    storage_options: Optional[Dict] = None,
    **config_kwargs,
) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:

    if use_auth_token != "deprecated":
        warnings.warn(
            "'use_auth_token' was deprecated in favor of 'token' in version 2.14.0 and will be removed in 3.0.0.\n"
            f"You can remove this warning by passing 'token={use_auth_token}' instead.",
            FutureWarning,
        )
        token = use_auth_token
    if ignore_verifications != "deprecated":
        verification_mode = VerificationMode.NO_CHECKS if ignore_verifications else VerificationMode.ALL_CHECKS
        warnings.warn(
            "'ignore_verifications' was deprecated in favor of 'verification_mode' in version 2.9.1 and will be removed in 3.0.0.\n"
            f"You can remove this warning by passing 'verification_mode={verification_mode.value}' instead.",
            FutureWarning,
        )
    if task != "deprecated":
        warnings.warn(
            "'task' was deprecated in version 2.13.0 and will be removed in 3.0.0.\n",
            FutureWarning,
        )
    else:
        task = None
    if data_files is not None and not data_files:
        raise ValueError(f"Empty 'data_files': '{data_files}'. It should be either non-empty or None (default).")
    if Path(path, config.DATASET_STATE_JSON_FILENAME).exists():
        raise ValueError(
            "You are trying to load a dataset that was saved using `save_to_disk`. "
            "Please use `load_from_disk` instead."
        )

    if streaming and num_proc is not None:
        raise NotImplementedError(
            "Loading a streaming dataset in parallel with `num_proc` is not implemented. "
            "To parallelize streaming, you can wrap the dataset with a PyTorch DataLoader using `num_workers` > 1 instead."
        )

    download_mode = DownloadMode(download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS)
    verification_mode = VerificationMode(
        (verification_mode or VerificationMode.BASIC_CHECKS) if not save_infos else VerificationMode.ALL_CHECKS
    )

    # Create a dataset builder
    print("******************* Create a dataset builder *******************")
    builder_instance = my_load_dataset_builder(
        path=path,
        name=name,
        data_dir=data_dir,
        data_files=data_files,
        cache_dir=cache_dir,
        features=features,
        download_config=download_config,
        download_mode=download_mode,
        revision=revision,
        token=token,
        storage_options=storage_options,
        **config_kwargs,
    )

    # Return iterable dataset in case of streaming
    if streaming:
        return builder_instance.as_streaming_dataset(split=split)

    # Some datasets are already processed on the HF google storage
    # Don't try downloading from Google storage for the packaged datasets as text, json, csv or pandas
    try_from_hf_gcs = path not in _PACKAGED_DATASETS_MODULES

    # Download and prepare data
    print("******************* Download and prepare data *******************")
    builder_instance.download_and_prepare(
        download_config=download_config,
        download_mode=download_mode,
        verification_mode=verification_mode,
        try_from_hf_gcs=try_from_hf_gcs,
        num_proc=num_proc,
        storage_options=storage_options,
    )

    # Build dataset for splits
    print("******************* Build dataset for splits *******************")
    keep_in_memory = (
        keep_in_memory if keep_in_memory is not None else is_small_dataset(builder_instance.info.dataset_size)
    )
    ds = builder_instance.as_dataset(split=split, verification_mode=verification_mode, in_memory=keep_in_memory)
    # Rename and cast features to match task schema
    if task is not None:
        # To avoid issuing the same warning twice
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            ds = ds.prepare_for_task(task)
    if save_infos:
        builder_instance._save_infos()

    return ds
