# Fact

Evidence-status fact-checking backend. Full README lands at the end of the build; see `.env.example` for configuration.

Quick start (mock mode, no API keys):

```bash
pip install -e '.[dev]'
cp .env.example .env
uvicorn app.main:app --reload   # open http://localhost:8000/
pytest
```
