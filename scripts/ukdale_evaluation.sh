args=("$@")
echo "app/model,mae_neuralnet,mae_minizinc","sae_neuralnet","sae_minizinc" > ${args[1]}.csv
bash scripts/runs_${args[0]}.sh >> ${args[1]}.csv
