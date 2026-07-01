#!/bin/bash
set -e

echo "Waiting for Weaviate..."
until curl -sf "http://${WEAVIATE_HOST:-localhost}:8080/v1/.well-known/ready" > /dev/null 2>&1; do
  sleep 2
done
echo "Weaviate is ready"

python -c "
import weaviate
client = weaviate.connect_to_local(host='${WEAVIATE_HOST:-localhost}')
if not client.collections.exists('Assessment'):
    print('No data found. Running embedder...')
    client.close()
    import embedder
    embedder.main()
else:
    count = client.collections.get('Assessment').aggregate.over_all(total_count=True).total_count
    print(f'Weaviate has {count} assessments')
    client.close()
"

echo "Starting API server..."
exec uvicorn app:app --host 0.0.0.0 --port 8000
