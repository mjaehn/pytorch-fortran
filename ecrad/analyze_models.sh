#!/bin/bash

python /users/juckerj/.local/lib/python3.9/site-packages/tensorflow/python/tools/saved_model_cli.py show --dir '../models/tmp_NN_othersave/' --tag_set serve --signature_def serving_default

