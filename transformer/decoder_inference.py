from typing import Any

import numpy as np
import onnxruntime as ort

from gtrs.model import TabEncodedSymbol
from gtrs.simple_logging import eprint
from gtrs.transformer.configs import TabConfig
from gtrs.type_definitions import NDArray


class TabScoreDecoder:
    def __init__(
        self,
        transformer: ort.InferenceSession,
        fp16: bool,
        use_gpu: bool,
        config: TabConfig,
    ) -> None:
        self.config = config
        self.net = transformer
        self.io_binding = self.net.io_binding()
        self.max_seq_len = config.max_seq_len
        self.eos_token = config.eos_token

        self.inv_rhythm_vocab = {v: k for k, v in config.rhythm_vocab.items()}
        self.inv_tab_vocab = {v: k for k, v in config.tab_vocab.items()}
        self.inv_technique_vocab = {v: k for k, v in config.technique_vocab.items()}
        self.inv_articulation_vocab = {v: k for k, v in config.articulation_vocab.items()}
        self.inv_position_vocab = {v: k for k, v in config.position_vocab.items()}

        self.fp16 = fp16
        self.use_gpu = use_gpu
        self.device_id = 0
        self.output_names = [
            "out_rhythms",
            "out_tabs",
            "out_techniques",
            "out_positions",
            "out_articulations",
            "attention",
        ]

    def generate(
        self,
        start_tokens: NDArray,
        nonote_tokens: NDArray,
        **kwargs: Any,
    ) -> list[TabEncodedSymbol]:
        num_dims = len(start_tokens.shape)
        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        out_rhythm = start_tokens
        out_tab = nonote_tokens
        out_technique = nonote_tokens
        out_articulations = nonote_tokens
        cache, kv_input_names, kv_output_names = self._init_cache()
        output_names = self.output_names + kv_output_names
        context = kwargs["context"]
        context_reduced = kwargs["context"][:, :1]

        symbols: list[TabEncodedSymbol] = []

        for step in range(self.max_seq_len):
            x_technique = out_technique[:, -1:]
            x_tab = out_tab[:, -1:]
            x_rhythm = out_rhythm[:, -1:]
            x_articulations = out_articulations[:, -1:]

            context = context if step == 0 else context_reduced

            self.io_binding.bind_cpu_input("rhythms", x_rhythm)
            self.io_binding.bind_cpu_input("tabs", x_tab)
            self.io_binding.bind_cpu_input("techniques", x_technique)
            self.io_binding.bind_cpu_input("articulations", x_articulations)
            self.io_binding.bind_cpu_input("context", context)
            self.io_binding.bind_cpu_input("cache_len", np.array([step], dtype=np.int64))
            for name, cache_val in zip(kv_input_names, cache, strict=True):
                self.io_binding.bind_ortvalue_input(name, cache_val)

            for name in output_names:
                self.io_binding.bind_output(
                    name, "cuda" if self.use_gpu else "cpu", self.device_id
                )

            self.net.run_with_iobinding(iobinding=self.io_binding)

            outputs = self.io_binding.get_outputs()
            cache = outputs[6:]

            rhythmsp = outputs[0].numpy()
            tabsp = outputs[1].numpy()
            techniquesp = outputs[2].numpy()
            positionsp = outputs[3].numpy()
            articulationsp = outputs[4].numpy()
            attention = outputs[5].numpy()

            rhythm_sample = np.array([[rhythmsp[:, -1, :].argmax()]])
            tab_sample = np.array([[tabsp[:, -1, :].argmax()]])
            technique_sample = np.array([[techniquesp[:, -1, :].argmax()]])
            articulation_sample = np.array([[articulationsp[:, -1, :].argmax()]])
            position_sample = np.array([[positionsp[:, -1, :].argmax()]])

            rhythm_probs = rhythmsp[:, -1, :]
            tab_probs = tabsp[:, -1, :]
            technique_probs = techniquesp[:, -1, :]
            articulation_probs = articulationsp[:, -1, :]
            position_probs = positionsp[:, -1, :]

            confidence = float(
                (
                    rhythm_probs.max()
                    + tab_probs.max()
                    + technique_probs.max()
                    + articulation_probs.max()
                    + position_probs.max()
                )
                / 5
            )

            technique_token = _detokenize(technique_sample, self.inv_technique_vocab)
            tab_token = _detokenize(tab_sample, self.inv_tab_vocab)
            rhythm_token = _detokenize(rhythm_sample, self.inv_rhythm_vocab)
            articulation_token = _detokenize(articulation_sample, self.inv_articulation_vocab)
            position_token = _detokenize(position_sample, self.inv_position_vocab)

            if rhythm_sample[0][0] == self.eos_token:
                break

            symbol = TabEncodedSymbol(
                rhythm=rhythm_token[0],
                tab=tab_token[0],
                technique=technique_token[0],
                articulation=articulation_token[0],
                position=position_token[0],
                confidence=confidence,
            )
            symbols.append(symbol)

            out_technique = np.concatenate((out_technique, technique_sample), axis=-1)
            out_tab = np.concatenate((out_tab, tab_sample), axis=-1)
            out_rhythm = np.concatenate((out_rhythm, rhythm_sample), axis=-1)
            out_articulations = np.concatenate((out_articulations, articulation_sample), axis=-1)

        return symbols

    def _init_cache(self, cache_len: int = 0) -> tuple[list[NDArray], list[str], list[str]]:
        cache = []
        input_names = []
        output_names = []
        heads = self.config.decoder_heads
        head_dim = self.config.decoder_dim // heads
        for i in range(self.config.decoder_depth * 4):
            dtype = np.float16 if self.fp16 else np.float32
            cache.append(
                ort.OrtValue.ortvalue_from_numpy(
                    np.zeros((1, heads, cache_len, head_dim), dtype=dtype),
                    "cuda" if self.use_gpu else "cpu",
                    self.device_id,
                )
            )
            input_names.append(f"cache_in{i}")
            output_names.append(f"cache_out{i}")
        return cache, input_names, output_names


def _detokenize(tokens: NDArray, vocab: dict[int, str]) -> list[str]:
    toks = [vocab[tok.item()] for tok in tokens]
    toks = [t for t in toks if t not in ("PAD", "BOS", "EOS")]
    return toks


def get_tab_decoder(config: TabConfig) -> TabScoreDecoder:
    use_gpu = False
    if config.use_gpu_inference:
        try:
            onnx_transformer = ort.InferenceSession(
                config.filepaths.decoder_path_fp16, providers=["CUDAExecutionProvider"]
            )
            fp16 = True
            if "CUDAExecutionProvider" in onnx_transformer.get_providers():
                use_gpu = True
            else:
                eprint("Onnxruntime not using GPU, falling back to CPU")
        except Exception as ex:
            eprint(ex)
            eprint("Going on without GPU support")
            onnx_transformer = ort.InferenceSession(config.filepaths.decoder_path_fp16)
            fp16 = True
    else:
        onnx_transformer = ort.InferenceSession(config.filepaths.decoder_path)
        fp16 = False

    return TabScoreDecoder(onnx_transformer, fp16, use_gpu, config=config)