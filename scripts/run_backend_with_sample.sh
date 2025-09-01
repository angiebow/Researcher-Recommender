
#!/usr/bin/env bash
cd "$(dirname "$0")/.."/backend
export DATA_PATH=../data/processed_data.sample.csv
uvicorn app.main:app --reload
