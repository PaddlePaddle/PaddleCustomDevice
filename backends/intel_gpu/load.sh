


PaddleDev=$(dirname $(dirname `pwd`))

echo $d
export PYTHONPATH=$PYTHONPATH:${PaddleDev}/python/tests/


comp="dnnl tbb compiler dpl"

for item in $comp;
do

	P="${HOME}/intel/oneapi/$item/latest/env/vars.sh"
        echo "$P"
        source $P
done


