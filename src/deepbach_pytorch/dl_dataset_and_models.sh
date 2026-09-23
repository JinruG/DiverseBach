#!/usr/bin/env bash
wget https://www.dropbox.com/s/iuz4ml857ycyyat/deepbach_pytorch_resources.tar.gz
tar xvfz deepbach_pytorch_resources.tar.gz
# move resources into the package folder: weight files land directly next to
# deepBach.py, matching the package-default models directory
mv resources/dataset_cache DatasetManager/dataset_cache
mv resources/models/* ./
rm -R resources