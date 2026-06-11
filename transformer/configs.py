import json
import os
from typing import Any

from gtrs.transformer import TabVocabulary

workspace = os.path.join(os.path.dirname(__file__))
root_dir = os.getcwd()


class TabFilePaths:
    def __init__(self) -> None:
        model_name = "tab_transformer_v1"
        self.encoder_path = os.path.join(workspace, f"encoder_{model_name}.onnx")
        self.decoder_path = os.path.join(workspace, f"decoder_{model_name}.onnx")
        self.encoder_path_fp16 = os.path.join(workspace, f"encoder_{model_name}_fp16.onnx")
        self.decoder_path_fp16 = os.path.join(workspace, f"decoder_{model_name}_fp16.onnx")
        self.checkpoint = os.path.join(
            root_dir, "training", "architecture", "transformer", f"{model_name}.pth"
        )
        self.rhythm_tokenizer = os.path.join(workspace, "tokenizer_rhythm.json")
        self.technique_tokenizer = os.path.join(workspace, "tokenizer_technique.json")
        self.tab_tokenizer = os.path.join(workspace, "tokenizer_tab.json")
        self.articulation_tokenizer = os.path.join(workspace, "tokenizer_articulation.json")

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "rhythm_tokenizer": self.rhythm_tokenizer,
            "technique_tokenizer": self.technique_tokenizer,
            "tab_tokenizer": self.tab_tokenizer,
            "articulation_tokenizer": self.articulation_tokenizer,
        }


class TabDecoderArgs:
    def __init__(self) -> None:
        self.attn_on_attn = True
        self.cross_attend = True
        self.ff_glu = True
        self.rel_pos_bias = False
        self.use_scalenorm = False
        self.attn_dropout = 0.1
        self.ff_dropout = 0.1
        self.layer_dropout = 0.1

    def to_dict(self) -> dict[str, Any]:
        return {
            "attn_on_attn": self.attn_on_attn,
            "cross_attend": self.cross_attend,
            "ff_glu": self.ff_glu,
            "rel_pos_bias": self.rel_pos_bias,
            "use_scalenorm": self.use_scalenorm,
            "attn_dropout": self.attn_dropout,
            "ff_dropout": self.ff_dropout,
            "layer_dropout": self.layer_dropout,
        }


class TabConfig:
    def __init__(self) -> None:
        self.vocab = TabVocabulary()
        self.filepaths = TabFilePaths()
        self.channels = 1
        self.patch_size = 16
        self.max_height = 256
        self.max_width = 1280
        self.max_seq_len = 608
        self.pad_token = 0
        self.bos_token = 1
        self.eos_token = 2
        self.nonote_token = 0
        self.num_rhythm_tokens = len(self.vocab.rhythm)
        self.num_tab_tokens = len(self.vocab.tab)
        self.num_technique_tokens = len(self.vocab.technique)
        self.num_articulation_tokens = len(self.vocab.articulation)
        self.num_position_tokens = len(self.vocab.position)
        self.encoder_structure = "convnext"
        self.encoder_depth = 8
        self.backbone_layers = [3, 4, 6, 3]
        self.encoder_dim = 512
        self.encoder_h_dim = self.encoder_dim // 3
        self.encoder_heads = 8
        self.decoder_dim = self.encoder_dim
        self.decoder_depth = 8
        self.decoder_heads = 8
        self.decoder_args = TabDecoderArgs()
        self.technique_vocab = self.vocab.technique
        self.tab_vocab = self.vocab.tab
        self.rhythm_vocab = self.vocab.rhythm
        self.articulation_vocab = self.vocab.articulation
        self.position_vocab = self.vocab.position
        self.use_gpu_inference = True
        self.scheduled_sampling_start_prob = 1.0
        self.scheduled_sampling_end_prob = 0.7
        self.scheduled_sampling_decay_steps = 20000

    def to_dict(self) -> dict[str, Any]:
        return {
            "filepaths": self.filepaths.to_dict(),
            "channels": self.channels,
            "patch_size": self.patch_size,
            "max_height": self.max_height,
            "max_width": self.max_width,
            "max_seq_len": self.max_seq_len,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "nonote_token": self.nonote_token,
            "encoder_structure": self.encoder_structure,
            "encoder_depth": self.encoder_depth,
            "backbone_layers": self.backbone_layers,
            "encoder_dim": self.encoder_dim,
            "encoder_heads": self.encoder_heads,
            "num_rhythm_tokens": self.num_rhythm_tokens,
            "num_tab_tokens": self.num_tab_tokens,
            "num_technique_tokens": self.num_technique_tokens,
            "decoder_dim": self.decoder_dim,
            "decoder_depth": self.decoder_depth,
            "decoder_heads": self.decoder_heads,
            "decoder_args": self.decoder_args.to_dict(),
        }


default_tab_config = TabConfig()