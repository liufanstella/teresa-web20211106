# 遠離能量吸血鬼 — YouTube Shorts 製作紀錄

頻道：奕凡靈性探索宇宙（I Fan Spiritual Exploration Universe）
系列：遠離能量吸血鬼，共 10 集，每集約 48～65 秒，直式 1080×1920、30 fps。
旁白稿來源：`遠離能量吸血鬼 YouTube Shorts 十集完整旁白稿`（docx）。

## 目前定案（2026-11-07）

| 集數 | 封面 | 配音 | 旁白／字幕 |
|---|---|---|---|
| 1 | `images/ep01.jpg` | `voice/ep01.mp3`（Jessica，11/07 07:18 版） | docx 第01集 |
| 2 | `images/ep02.jpg` | `voice/ep02.mp3`（Jessica） | docx 第02集 |
| 3 | `images/ep03.jpg` | `voice/ep03.mp3`（Jessica） | docx 第03集 |
| 4 | `images/ep04.jpg` | `voice/ep04.mp3`（Jessica） | docx 第04集 |
| 5 | `images/ep05.jpg`（封面：能量防護罩） | `voice/ep05.mp3`（Jessica） | docx 第05集「停止解釋」 |
| 6 | `images/ep06.jpg`（封面：從內耗到充電） | `voice/ep06.mp3`（Jessica） | docx 第06集「遠離負能量」 |
| 7 | `images/ep07.jpg`（封面：學會說「不」） | `voice/ep07.mp3`（Jessica） | docx 第07集「三個跡象」 |
| 8 | `images/ep08.jpg`（封面：情緒排毒） | `voice/ep08.mp3`（Jessica） | docx 第08集「斷捨離」 |
| 9 | `images/ep09.jpg`（封面：把注意力還給自己） | `voice/ep09.mp3`（Jessica） | docx 第09集「每天三分鐘」 |
| 10 | `images/ep10.jpg`（封面：能量主權者） | `voice/ep10.mp3`（Jessica） | docx 第10集「靈性覺醒」 |

### 使用者的決定（之後請照做）

- **配音一律用使用者提供的 ElevenLabs Jessica 錄音**（Jessica – Calm, Friendly）。每份錄音開頭都先念「第N集。（集名）」，這段沒有字幕，因為封面上已經有標題。
- **封面只用使用者上傳的圖**，不要自己生成或設計封面。
- **第5～10集的封面標題和 docx 旁白主題不同**。使用者選擇保留錄音（docx 舊稿），封面照舊，因此標題和內容不一致。依封面重寫的旁白稿（未採用）留在 commit `0b4701e` 的 `shorts/scripts.py`。
- **背景音樂**：使用者提供的 `music/bgm.mp3`，十集共用。音樂放在旁白下方約 17 dB，開頭淡入 1.5 秒、結尾淡出 3.5 秒。
- **字幕**：中文在上、英文在下。行距加大，每一行是獨立的字幕框，中文行距 106 px、字級 56；英文用斜體 Bitstream Charter、字級 38。英文翻譯在 `scripts.py`。
- **待確認**：第4集約 17～20 秒，語音辨識聽到「也許可能提醒我們」，稿子是「有時可能」。若錄音確實念「也許」，把 `scripts.py` 第4集那一句改成一致即可。

## 檔案

- `scripts.py`：每集的標題、封面路徑和旁白，旁白是 `(中文, English)` 逐句列表。
- `build.py`：產生影片，包括配音對時、混音、字幕和畫面。
- `music.py`：備用的合成柔和管弦背景音樂，只在沒有 `music/` 檔案時才使用。
- `voice/epNN.mp3`：每集的 Jessica 錄音。
- `voice/reference.wav` / `reference.txt`：仿聲參考片段（取自第2集錄音），只在某集沒有錄音時才用。
- `images/epNN.jpg`：封面。
- `output/`：成品 mp4 與 srt（不進 git）。

## 怎麼重新產生

```bash
cd shorts
python3 build.py            # 全部 10 集
python3 build.py 3 7        # 只做指定集數
```

需要 `ffmpeg`，以及 Python 套件 `numpy scipy soundfile`。
配音的選用順序：`voice/epNN.(wav|mp3|m4a)` 錄音 → 環境變數 `ELEVENLABS_API_KEY`（Jessica，voice id `cgSgspJ2msm6clMCkdW9`）→ `voice/reference.wav` 仿聲（sherpa-onnx ZipVoice）→ 離線普通話女聲。只有沒有錄音時才需要 TTS 模型，放在 `TTS_MODEL_DIR`（預設 `shorts/models/`，不進 git）。

### 換錄音或改字句

1. 把新錄音存成 `voice/epNN.mp3`。錄音開頭要念「第N集。集名」。
2. 如果錄音念的稿子和 `scripts.py` 不同，先把 `scripts.py` 改成和錄音一致。字幕是依錄音裡的停頓，一句一句對到 `lines` 上的，所以句數和內容必須相符。
3. 執行 `python3 build.py N`。
4. 檢查：用語音辨識（例如 sherpa-onnx paraformer-zh）比對錄音和稿子。

### 輸出規格

- 畫面：封面緩慢推近（7%），開頭和結尾各淡入淡出 0.6／1.0 秒。
- 聲音：開頭留 0.8 秒、結尾留 2.2 秒；整體響度正規化到 −14 LUFS（YouTube 標準），AAC 192k、48 kHz。
- 字幕燒進畫面，另外輸出 `output/epNN.srt`，可上傳 YouTube 當 CC 字幕。
