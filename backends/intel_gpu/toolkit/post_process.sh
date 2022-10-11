
mpd="../../../Paddle"


if [ ! -d $mpd ]; then
     echo "PaddlePaddle dir doesn't exist" 
     exit
fi

cd $mpd

if [ ! -f "CMakeLists.txt" ]; then
   echo "$mpd   is empty .. update paddlepaddle src && git submodule update"
   exit
fi




ulimit -n 8192

last_commit=`git log | head -n 1 | cut -d ' ' -f 2`

out_dir+="_"
out_dir+="$last_commit"

if [ ! -d "$out_dir" ]; then
    mkdir $out_dir
fi

cd $out_dir

if [ $? -eq 0 ]; then

     if [ $OPT_CLEAR -eq 1 ]; then
          rm -rf *
     fi	     
fi

if [ $OPT_SKIPCMAKE -eq 0 ]; then
  cmake_cmd="cmake $cmake_cmd $CMAKE_EXTRA_ARGS .."
  echo $cmake_cmd > diag.txt
  $cmake_cmd
fi

if [ $OPT_VERBOSE -eq 1 ]; then
make VERBOSE=1 -j $OPT_CORES
else 
make  -j $OPT_CORES
fi
#make VERBOSE=1 

if [ $? -eq 0 ]; then
 echo "PYTHONPATH=`pwd`/$out_dir/python" > load_PYTHON.sh
fi
