#!/bin/bash
`python3 -m venv env`
source env/bin/activate
pip install numpy
pip install scipy
pip install pandas
pip freeze -> requirements.txt