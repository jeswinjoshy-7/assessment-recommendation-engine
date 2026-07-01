#!/bin/bash
set -e

HOST="${WEAVIATE_HOST:-localhost}"
echo "Waiting for Weaviate at $HOST:8080..."

for i in $(seq 1 30); do
  if curl -sf "http://$HOST:8080/v1/.well-known/ready" > /dev/null 2>&1; then
    echo "Weaviate is ready"
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "Weaviate not reachable after 60s. Starting anyway..."
  fi
  sleep 2
done

python -c "
import weaviate, os
host = os.getenv('WEAVIATE_HOST', 'localhost')
client = weaviate.connect_to_local(host=host)
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
