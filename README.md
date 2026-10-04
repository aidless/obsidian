# paperKB

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**This is not related to the Obsidian note-taking application.** It is a FAISS + SQLite
toolkit for searching a large paper knowledge base — the name refers to the author's
local toolchain, not the editor.

## Tools

| Script | Purpose |
|---|---|
| `kb_search.py` | search the index |
| `kb_health.py` | report index health and coverage |
| `dedup_and_reindex.py` | deduplicate and rebuild the index |
| `merge_to_disk.py` | merge index shards on disk |
| `smart_rerank.py` | rerank search results |

## Setup

```bash
pip install -r requirements.txt
python kb_health.py
```

## Deployment

See [CICD_DEPLOYMENT_CHECKLIST.md](CICD_DEPLOYMENT_CHECKLIST.md).

## License

MIT. See [LICENSE](LICENSE).
