"""
This script is used for automatic speech recognition (ASR) using the OpenAI Whisper model.
It loads the model and processor from the specified model ID, and sets up a pipeline for ASR.
The script also loads a validation dataset and a sample audio file for testing.

The pipeline is configured to return timestamps, and the script demonstrates how
to use the pipeline to process an audio file.

Parameters:
- DEVICE: Specifies the device to use for computations (CPU or GPU).
- TORCH_DTYPE: Specifies the data type for torch tensors.
- MODEL_ID: The identifier for the pre-trained model to use.
- FILENAME: The path to the audio file to process.

Dependencies:
- torch
- transformers
- datasets
"""
import argparse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset



def create_pipe():
    """
    This function creates an automatic speech recognition pipeline using the specified model, tokenizer, and feature extractor.

    Parameters:
    device (str): The device to use for computations (CPU or GPU).
    torch_dtype (torch.dtype): The data type for torch tensors.
    model_id (str): The identifier for the pre-trained model to use.

    Returns:
    pipe (pipeline): The automatic speech recognition pipeline.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    # if graphics card is very new, use flash_attention_2=True
    # model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id,
    #         torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,
    #         use_flash_attention_2=True)
    model.to(device)
    model = model.to_bettertransformer() # for older gpus

    processor = AutoProcessor.from_pretrained(model_id)

    pipe_ = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        # chunk_length_s=30,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe_


def get_sample(dataset_path="distil-whisper/librispeech_long", dataset_name="clean"):
    """
    This function loads a sample from the validation split of the specified dataset.

    Parameters:
    dataset_path (str): The path to the dataset.
    dataset_name (str): The name of the dataset.

    Returns:
    sample (dict): A dictionary containing the audio data of the first sample in the validation split.
    """
    dataset = load_dataset(dataset_path, dataset_name, split="validation")
    sample_ = dataset[0]["audio"]
    return sample_


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--filename", type=str, default=None)
    argparser.add_argument("--language", type=str, default="english")
    argparser.add_argument("--demo1", action="store_true")
    argparser.add_argument("--demo2", action="store_true")
    args = argparser.parse_args()

    pipe = create_pipe()
    generate_kwargs = {}
    if args.language:
        generate_kwargs["language"] = args.language

    if args.demo1:
        filename = get_sample()
    elif args.demo2:
        filename = r"sound/test.mp4"
    else:
        filename = args.filename
    if filename is None:
        raise ValueError("Please specify a filename or use the --demo1 or --demo2 flag.")
    result = pipe(filename, return_timestamps=True, generate_kwargs=generate_kwargs)
    # reuslt_chunks = [{'timestamp': (0.0, 27.58), 'text': '作詞・作曲・編曲 初音ミク踊ってるだけで退場 それをそっかそっかって言って'}, 
    #                  {'timestamp': (27.58, 33.26), 'text': 'お幸せについて討論 何が正義なんか って思う'}, 
    #                  {'timestamp': (33.26, 39.16), 'text': '名前くそにがむかんで それもいいないいなって思う'}, 
    #                  {'timestamp': (39.16, 47.14), 'text': 'テレスコープ越しの感情 廊下に全部詰めてんだ踊 ってない夜を知らない'}]
    reuslt_chunks = result["chunks"]
    # for chunk in reuslt_chunks:
    #     print(f'{chunk["timestamp"][0]} -> {chunk["timestamp"][1]}: {chunk["text"]}')
    for chunk in reuslt_chunks:
        start_min, start_sec = divmod(chunk["timestamp"][0], 60)
        end_min, end_sec = divmod(chunk["timestamp"][1], 60)
        print(f'{start_min:.0f}m {start_sec:.0f}s -> {end_min:.0f}m {end_sec:.0f}s: {chunk["text"]}')
