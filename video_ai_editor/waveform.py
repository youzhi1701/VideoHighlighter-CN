import os
import tempfile
import subprocess
import wave
import numpy as np

class WaveformVisualizer:
    """Extracts and stores waveform data for visualization"""
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.waveform_data = None  # List of (min_val, max_val) tuples
        self.duration = 0
        self.sample_rate = 44100
    
    def extract_waveform(self, num_points=1000):
        import os, tempfile, subprocess, wave
        import numpy as np

        fd, wav_file = tempfile.mkstemp(suffix=".wav")
        os.close(fd)  # IMPORTANT: don't keep the file handle open

        try:
            print(f"🎵 正在从以下文件提取音频：{self.video_path}")

            cmd = [
                "ffmpeg",
                "-y",
                "-i", self.video_path,
                "-map", "0:a:0",          # pick first audio stream explicitly
                "-vn",
                "-ac", "1",
                "-ar", str(self.sample_rate),
                "-c:a", "pcm_s16le",
                wav_file,
                "-hide_banner",
                "-loglevel", "error",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ FFmpeg 执行失败")
                print("标准错误：", result.stderr.strip())
                return None

            if not os.path.exists(wav_file) or os.path.getsize(wav_file) < 44:
                print("❌ WAV 输出不存在或过小（可能没有音频，或 FFmpeg 写入失败）")
                return None

            with wave.open(wav_file, "rb") as wf:
                rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16)

            if audio.size == 0 or rate <= 0:
                print("❌ 未解码出音频样本")
                return None

            self.duration = audio.size / rate

            step = max(1, audio.size // num_points)
            waveform = []
            for i in range(0, audio.size, step):
                chunk = audio[i:i + step]
                if chunk.size:
                    rms = float(np.sqrt(np.mean((chunk.astype(np.float64) / 32768.0) ** 2)))
                    waveform.append((float(chunk.min()) / 32768.0,
                                     float(chunk.max()) / 32768.0, rms))

            self.waveform_data = waveform
            print(f"✅ 波形提取完成：{len(waveform)} 个点，时长={self.duration:.2f} 秒")
            return waveform

        except Exception as e:
            print(f"❌ 波形提取出错：{e}")
            import traceback; traceback.print_exc()
            return None

        finally:
            try:
                if os.path.exists(wav_file):
                    os.remove(wav_file)
            except:
                pass
