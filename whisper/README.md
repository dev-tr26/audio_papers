# Whisper-Small LoRA Fine-tuning on Kathbath Gujarati

Fine-tuning OpenAI's Whisper-Small on the AI4Bharat Kathbath Gujarati ASR dataset using LoRA (Low-Rank Adaptation). Targets Kaggle 2x T4 GPUs with fp16 training.

---

## Model

| Property | Value |
|----------|-------|
| Base model | `openai/whisper-small` |
| Method | LoRA (PEFT) |
| Language | Gujarati (`gu`) |
| Task | Transcription |
| Dataset | `ai4bharat/kathbath` |
| Precision | fp16 |
| Target hardware | Kaggle 2x T4 (15 GiB each) |

---

## LoRA Config

| Param | Value |
|-------|-------|
| `r` | 32 |
| `lora_alpha` | 64 |
| `dropout` | 0.05 |
| `bias` | none |
| Target modules | `q_proj`, `v_proj`, `k_proj`, `out_proj`, `fc1`, `fc2` |

---

## Pipeline Overview

```
1. Load tokenizer, feature extractor, processor  (Whisper-Small)
2. Load dataset                                   (ai4bharat/kathbath, gu)
3. Split                                          (train[:23%] / valid / test)
4. Vocabulary analysis                            (word + char frequency plots)
5. Audio preprocessing                            (resample → 16kHz, truncate/pad, log-mel)
6. Tokenizer sanity check                         (5 random samples encode→decode)
7. Apply LoRA to Whisper-Small
8. Train with Seq2SeqTrainer                      (fp16, cosine LR, early stopping)
9. Plot training curves                           (loss, WER, CER)
10. Visualize spectrograms
11. Evaluate on test set                          (WER, CER)
12. Save merged model + PyTorch checkpoint
13. Export to ONNX                                (edge deployment)
14. Download model                                (4 format options)
15. Log everything to W&B
```

---

## Setup

### Requirements

```bash
pip install transformers datasets peft evaluate jiwer rouge_score \
            librosa soundfile wandb accelerate torchaudio \
            onnx onnxruntime optimum[onnxruntime]
```

### HuggingFace Access

The Kathbath dataset is gated. You need to:

1. Create an account at https://huggingface.co
2. Request access at https://huggingface.co/datasets/ai4bharat/kathbath
3. Generate a token at https://huggingface.co/settings/tokens
4. On Kaggle, store the token as a secret named `HF_TOKEN_3` and load it:

```python
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

hf_token = UserSecretsClient().get_secret("HF_TOKEN_3")
login(token=hf_token)
```

### W&B

```python
import wandb
wandb.login()  # paste API key from https://wandb.ai/authorize
```

---

## Hyperparameters

All hyperparameters are in the `HParams` dataclass at the top of the notebook. Key ones:

```python
lora_r              = 32
lora_alpha          = 64
learning_rate       = 1e-4
num_train_epochs    = 8
per_device_train_bs = 8
gradient_accumulation = 2
warmup_steps        = 300
early_stopping_patience = 4
```

---

## Metrics

| Metric | Description |
|--------|-------------|
| WER | Word Error Rate (primary, lower is better) |
| CER | Character Error Rate (lower is better) |

---

## Saved Outputs

| Path | Contents |
|------|----------|
| `./whisper-small-gu-lora/` | LoRA adapter weights |
| `./whisper-small-gu-lora/best_merged/` | Merged HF model (safetensors) |
| `./whisper-small-gu-lora/best_merged/checkpoint.pt` | Raw PyTorch state dict |
| `./whisper-small-gu-onnx/` | ONNX model for edge deployment |

---

## Loading the Model

### Merged HF model

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("./whisper-small-gu-lora/best_merged")
model = WhisperForConditionalGeneration.from_pretrained("./whisper-small-gu-lora/best_merged")
```

### LoRA adapter only

```python
from transformers import WhisperForConditionalGeneration
from peft import PeftModel

base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = PeftModel.from_pretrained(base, "./whisper-small-gu-lora")
model = model.merge_and_unload()
```

### ONNX (CPU / edge)

```python
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import pipeline, WhisperProcessor

processor = WhisperProcessor.from_pretrained("./whisper-small-gu-onnx")
ort_model = ORTModelForSpeechSeq2Seq.from_pretrained("./whisper-small-gu-onnx")
pipe = pipeline("automatic-speech-recognition", model=ort_model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor)
print(pipe("audio.wav")["text"])
```

---

## Inference

```python
import torch, librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "./whisper-small-gu-lora/best_merged"

processor = WhisperProcessor.from_pretrained(MODEL_DIR)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

def transcribe(audio_path: str) -> str:
    waveform, sr = librosa.load(audio_path, sr=16_000, mono=True)
    inputs = processor(waveform, sampling_rate=16_000, return_tensors="pt")
    input_features = inputs.input_features.to(DEVICE)
    forced_ids = processor.get_decoder_prompt_ids(language="Gujarati", task="transcribe")
    with torch.no_grad():
        ids = model.generate(input_features, forced_decoder_ids=forced_ids)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]

print(transcribe("sample.wav"))
```

---

## Author

Rudra — [github.com/dev-tr26](https://github.com/dev-tr26) | [huggingface.co/rtxtd](https://huggingface.co/rtxtd)