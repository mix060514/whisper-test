import tkinter as tk
import sounddevice as sd
import wavio
import threading
import torch
import numpy as np
import time
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


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



def record_audio(filename='recording.wav', stop_event=None, on_complete=None):
    # 錄製音頻
    fs = 44100  # 採樣率
    myrecording = []
    
    def callback(indata, frames, time, status):
        if status:
            print(status, flush=True)
        if stop_event.is_set():
            return sd.CallbackAbort  # 當 stop_event 被設置時，提前終止錄音
        myrecording.append(indata.copy())

    try:
        with sd.InputStream(samplerate=fs, channels=2, callback=callback):
            while not stop_event.is_set():
                sd.sleep(100)
    except sd.CallbackAbort:
        pass

    wav_data = np.concatenate(myrecording, axis=0)
    wavio.write(filename, wav_data, fs, sampwidth=2)
    if on_complete:
        on_complete()


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("語音轉文字")
        tk.Button(self.root, text="開始錄音", command=self.start_recording_thread).pack()
        tk.Button(self.root, text="停止錄音並轉寫", command=self.stop_and_transcribe).pack()
        self.show_timestamps_btn = tk.Button(self.root, text="顯示時間戳", command=self.toggle_show_timestamps)
        self.show_timestamps_btn.pack()
        self.show_timestamps = False
        self.text = tk.Text(self.root)
        self.text.pack()
        self.recording_thread = None
        self.filename = "sound/recording.wav"

    def start_recording_thread(self):
        self.stop_event = threading.Event()
        self.recording_thread = threading.Thread(target=record_audio,
                args=(self.filename, self.stop_event, self.on_recording_complete))
        self.recording_thread.start()

    def on_recording_complete(self):
        # 在錄音完成後更新 GUI，需要在主線程中執行
        self.root.after(0, self.stop_and_transcribe)

    def stop_and_transcribe(self):
        print("in stop_and_transcribe")
        if self.recording_thread and self.recording_thread.is_alive():
            self.stop_event.set()
            self.recording_thread.join()
        # 使用 WhisperLargeV3 模型進行轉寫
        pipe = create_pipe()
        result = pipe(self.filename, return_timestamps=True)
        result_chunks = result["chunks"]

        for chunk in result_chunks:
            if self.show_timestamps:
                start_min, start_sec = divmod(chunk["timestamp"][0], 60)
                end_min, end_sec = divmod(chunk["timestamp"][1], 60)
                self.text.insert(tk.END, f'{start_min:.0f}m {start_sec:.0f}s -> {end_min:.0f}m {end_sec:.0f}s: {chunk["text"]}\n')
            else: 
                self.text.insert(tk.END, f'{chunk["text"]}\n')

    def toggle_show_timestamps(self):
        self.show_timestamps = not self.show_timestamps
        btn_text = "隱藏時間戳" if self.show_timestamps else "顯示時間戳"
        self.show_timestamps_btn.config(text=btn_text)


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
