#!/bin/bash

python test_react.py  > log_react.txt 2>&1

python test_a1.py  > log_a1.txt 2>&1

python test_gemini.py  > log_gemini.txt 2>&1

