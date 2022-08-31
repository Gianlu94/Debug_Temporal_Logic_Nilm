set -e

mkdir -p /tmp/dvc_train_${app}*
rm -r /tmp/dvc_train_${app}*

export PYTHONPATH=deps/seq2point/src
args=("$@")

app=${args[0]}

singularity exec  --nv docker://tensorflow/tensorflow:2.8.0-gpu /usr/bin/python3 deps/seq2point/scripts/train_seq2point.py data/ukdale.npz ${app} dvc_train -v 

# take the fifth last (the best one due to earlystopping)
target=$(ls -1  /tmp/dvc_train_${app}* | tail -n 5 | head -n 1)

mkdir -p models
mv /tmp/dvc_train_${app}*/$target models/${app}.hdf5
rm -r /tmp/dvc_train_${app}*

