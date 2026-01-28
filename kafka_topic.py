docker exec -it kafka \
kafka-topics --create \
--topic video_jobs \
--bootstrap-server localhost:9092 \
--partitions 3 \
--replication-factor 1
