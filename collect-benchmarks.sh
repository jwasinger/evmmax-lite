#! /usr/bin/env bash

benchmark_dir="benchmarks-results"

echo "running go/asm arithmetic benchmarks"
(cd mont-arith && go test -run=^$ -bench=.*Asm | python3 ../format-geth-benchmark-output-as-csv.py) > $benchmark_dir/go-arith-benchmarks.csv
(cd mont-arith && go test -run=^$ -bench=.*Go | python3 ../format-geth-benchmark-output-as-csv.py) >> $benchmark_dir/go-arith-benchmarks.csv
(cd mont-arith && go test -run=^$ -bench=SetMod | python3 ../format-geth-benchmark-output-as-csv.py) >> $benchmark_dir/go-arith-benchmarks.csv

echo "running geth evm go benchmarks"
GETH_EVM=$(pwd)/go-ethereum/build/bin/evm
export GETH_EVM
(cd evm-benchmarks/evmmax_generator && python3 generate.py) > $benchmark_dir/geth-evm-go.csv
unset GETH_EVM

echo "running geth evm asm384 benchmarks"
GETH_EVM=$(pwd)/go-ethereum-asm384/build/bin/evm
export GETH_EVM
(cd evm-benchmarks/evmmax_generator && python3 generate.py MULMONTMAX 6) > $benchmark_dir/geth-evm-asm384.csv
(cd evm-benchmarks/evmmax_generator && python3 generate.py SUBMODMAX 6) >> $benchmark_dir/geth-evm-asm384.csv
(cd evm-benchmarks/evmmax_generator && python3 generate.py ADDMODMAX 6) >> $benchmark_dir/geth-evm-asm384.csv
unset GETH_EVM
