export PYTHONPATH=src 
app=kettle; /usr/bin/python3 scripts/evaluate.py ./data/ukdale.npz ${app} data/${app}_embed_test.npz --study-to-eval study_local --fit-solver or-tools  --fit-states 1 --skip-size 300  --fit-processes 8  --predict-solver or-tools  -p 1 -s 5   --seq-weights 1 1 --deviation 10 --train-reduction 10 --predict-reduction 30 --csv
app=microwave; /usr/bin/python3 scripts/evaluate.py ./data/ukdale.npz ${app} data/${app}_embed_test.npz --study-to-eval study_local  --fit-solver or-tools  --fit-states 1 --skip-size 300  --fit-processes 8  --predict-solver or-tools  -p 1 -s 5   --seq-weights 1 2 --deviation 10 --train-reduction 15 --predict-reduction 30 --csv
app=fridge; /usr/bin/python3 scripts/evaluate.py ./data/ukdale.npz ${app} data/${app}_embed_test.npz  --study-to-eval study_local --fit-solver or-tools  --fit-states 1 --skip-size 300  --fit-processes 8  --predict-solver or-tools  -p 1 -s 5   --seq-weights 1 0 --deviation 10 --train-reduction 15 --predict-reduction 30 --csv
app=dishwasher; /usr/bin/python3  scripts/evaluate.py ./data/ukdale.npz ${app} data/${app}_embed_test.npz  --study-to-eval study_local --fit-solver or-tools  --fit-states 3 --skip-size 1000  --fit-processes 8  --predict-solver or-tools  -p 1 -s 5   --seq-weights 1 2 --deviation 0.5 --train-reduction 10 --predict-reduction 20 --csv
app=washingmachine; /usr/bin/python3 scripts/evaluate.py ./data/ukdale.npz ${app} data/${app}_embed_test.npz --study-to-eval study_local --fit-solver or-tools  --fit-states 2 --skip-size 800  --fit-processes 8  --predict-solver or-tools  -p 1 -s 5   --seq-weights 1 1 --deviation 0.5 --train-reduction 15 --predict-reduction 20 --csv

