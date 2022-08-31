mkdir ukdale
pushd ukdale
wget -O ukdale.zip http://data.ukedc.rl.ac.uk/simplebrowse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/UK-DALE-FULL-disaggregated/ukdale.zip
unzip ukdale.zip
popd
mkdir -p data
/usr/bin/python3 deps/seq2point/scripts/ukdale_dataset_extraction.py ukdale data/ukdale.npz

rm -r ukdale
