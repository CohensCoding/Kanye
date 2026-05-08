## Kanye lyrics fetcher

Fetch Kanye West’s lyrics from Genius for a discography list and save:

- `kanye_lyrics.csv`
- `kanye_lyrics.json`
- `failed_tracks.txt` (tracks that couldn’t be found/scraped)

The script is resumable via a JSONL checkpoint at `.cache/kanye_lyrics_records.jsonl`.

### Setup

1. Create a virtualenv and install deps:

```bash
/usr/bin/python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If venv creation fails on your machine, you can also install deps to your user site-packages:

```bash
/usr/bin/python3 -m pip install --user -r requirements.txt
```

2. Create a `.env` file with your Genius token:

```bash
cp .env.example .env
```

Then edit `.env` and set `GENIUS_API_TOKEN`.

### Discography input format

By default the script expects `kanye_discography.md`.

It supports a few “one track per line” formats, for example:

```text
- Through the Wire | The College Dropout | 2003-09-30 | Single
- "Slow Jamz" — Late Registration (2004-11-10) [Feature/Single]
```

### Run

```bash
python scripts/fetch_kanye_lyrics.py --discography kanye_discography.md
```

Optional flags:

- `--out-csv path.csv`
- `--out-json path.json`
- `--checkpoint .cache/whatever.jsonl`
- `--failed failed_tracks.txt`
- `--min-delay 1.0 --max-delay 2.0`
- `--artist-hint "Kanye West"`

