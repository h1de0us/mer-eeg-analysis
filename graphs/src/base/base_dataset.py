import logging
import random
from typing import List

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from src.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
            self,
            index,
            config_parser: ConfigParser,
            limit=None,
            max_audio_length=None, # not in seconds
    ):
        self.config_parser = config_parser
        self.max_audio_length = max_audio_length
        self.limit = limit

        self._assert_index_is_valid(index)
        index = self._filter_records_from_dataset(index, max_audio_length, limit)
        # it's a good idea to sort index by audio length
        # It would be easier to write length-based batch samplers later
        index = self._sort_index(index)
        self._index: List[dict] = index

    # TODO: 
    def __getitem__(self, ind):
        return {
            "eeg": ...,
            "recording_len": ...,
            "duration": ... / self.config_parser["preprocessing"]["sr"],
            "audio_path": ...,
        }

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["recording_len"])

    def __len__(self):
        return len(self._index)

    # TODO: a method to load a raw waveform
    def load_raw_recording(self, path):
        pass

    # TODO: a method to load a raw waveform and convert it to spectrogram
    def convert_recording_into_spec(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor


    @staticmethod
    def _filter_records_from_dataset(
            index: list, max_audio_length, limit
    ) -> list:
        initial_size = len(index)
        if max_audio_length is not None:
            exceeds_audio_length = np.array([el["recording_len"] for el in index]) >= max_audio_length
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = False

        initial_size = len(index)

        records_to_filter = exceeds_audio_length

        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total}({_total / initial_size:.1%}) records  from dataset"
            )

        if limit is not None:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        return index

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "recording_len" in entry, (
                "Each dataset item should include field 'recording_len'"
                " - duration of eeg recording (in seconds)."
            )
            assert "path" in entry, (
                "Each dataset item should include field 'path'" " - path to eeg file."
            )
            assert "valence" in entry, (
                "Each dataset item should include field 'valence'"
                " - valence related to the current stimulus."
            )
            assert "arousal" in entry, (
                "Each dataset item should include field 'arousal'"
                " - arousal related to the current stimulus."
            )