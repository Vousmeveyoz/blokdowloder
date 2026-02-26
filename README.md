# 🎵 SoundCloud → OGG Downloader

Download lagu dari SoundCloud dan otomatis convert ke format **OGG** dengan nama file unik.

**Contoh output:**
```
Kita Usahakan Lagi - Batas_cmm3napev200ctumheg2y4cdl.ogg
```

---

## 📁 Struktur Project

```
soundcloud-downloader/
├── src/
│   ├── __init__.py           # Package exports
│   ├── downloader.py         # Download audio dari SoundCloud
│   ├── converter.py          # Convert MP3 → OGG
│   └── filename_generator.py # Generate nama file unik
├── main.py                   # Entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Requirements

- Python 3.9+
- ffmpeg (wajib terinstall di sistem)

### Install ffmpeg

**Windows:**
```bash
winget install ffmpeg
# atau download dari https://ffmpeg.org/download.html
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

---

## 🚀 Setup & Install

```bash
# 1. Clone / download project
cd soundcloud-downloader

# 2. Buat virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🎧 Cara Pakai

### Basic usage
```bash
python main.py https://soundcloud.com/artist/track-name
```

### Dengan opsi tambahan
```bash
# Simpan MP3 sementara (tidak dihapus setelah convert)
python main.py https://soundcloud.com/artist/track-name --keep-source

# Custom output dan temp directory
python main.py https://soundcloud.com/artist/track-name --output-dir downloads --temp-dir tmp
```

### Contoh output di terminal
```
=======================================================
  🎵  SoundCloud → OGG Downloader
=======================================================
  URL        : https://soundcloud.com/...
  Output dir : output/
=======================================================

[1/3] Downloading from SoundCloud...
  Downloaded  : Kita Usahakan Lagi - Batas

[2/3] Generating unique filename...
  Filename    : Kita Usahakan Lagi - Batas_cmm3napev200ctumheg2y4cdl.ogg

[3/3] Converting to OGG...
  Saved to    : output/Kita Usahakan Lagi - Batas_cmm3napev200ctumheg2y4cdl.ogg

=======================================================
  Done! Enjoy your music.
=======================================================
```

---

## 🛠️ Options

| Flag | Default | Deskripsi |
|---|---|---|
| `url` | *(required)* | SoundCloud track URL |
| `--keep-source` | `False` | Jangan hapus file MP3 sementara |
| `--output-dir` | `output/` | Folder output file OGG |
| `--temp-dir` | `temp/` | Folder file sementara |
