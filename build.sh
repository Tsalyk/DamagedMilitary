#!/bin/bash
docker compose -f docker-compose.yml up

npm install --from-lock-json
npm audit fix
