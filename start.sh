#!/bin/bash
uvicorn ingredients_classifier_api:app --host 0.0.0.0 --port $PORT
