
comp="dnnl tbb compiler"

for item in $comp;
do

	P="${HOME}/intel/oneapi/$item/latest/env/vars.sh"
        echo "$P"
        source $P
done


