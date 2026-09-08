# Contributing

## Workflow

1. Create a branch from `main`: `git switch -c feature/short-description`.
2. Keep changes focused and use Conventional Commits, for example
   `feat(retrieval): add weighted reciprocal-rank fusion`.
3. Run the offline tests before opening a pull request.
4. Run the smoke benchmark for changes to routing, retrieval, grading,
   generation, verification, cache, or conversation history.
5. Open a pull request and describe validation evidence and rollout risk.

## Local checks

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts\evaluate.py --profile smoke --production-mode --no-gate
```

The benchmark requires private files under `data/raw` plus valid OpenAI and
Qdrant credentials. Generated reports remain local under `reports/`.

## Security

Never commit `.env`, OpenAI/Qdrant credentials, Supabase service-role keys, JWT
secrets, private datasets, or benchmark reports. The Supabase anon key is a
public browser credential; Row Level Security must still be enabled and tested.
