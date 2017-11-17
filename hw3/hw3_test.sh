#!/bin/bash
wget 'https://www.dropbox.com/s/a5gl3rmi7ltccs2/model.h5'
python3 test.py $1 $2
