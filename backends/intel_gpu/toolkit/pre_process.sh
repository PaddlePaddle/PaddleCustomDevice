# apt-get install make patchelf libssl-dev libz-dev
# git clone https://github.com/PaddlePaddle/Paddle.git 
# cd Paddle
# pip install --user -r python/requirements.txt
# cd ..

ulimit -n 8192


usage() {

 echo "Usage : ...."
 echo "--jcores=1"
 echo "--verbose     => make VERBOSE=1"
 echo "--skipcmake  => skip cmake "
 echo "--clear  => rm -rf * build "

 exit;

}

OPT_DEBUG=0
OPT_RELEASE=0
CMAKE_EXTRA_ARGS="-DWITH_INFERENCE_API_TEST=ON -DWITH_TESTING=ON"
OPT_SKIPCMAKE=0
OPT_CORES=20
OPT_VERBOSE=0
OPT_CLEAR=0
for i in "$@"; do
  case $i in
    -j=*|--jcores=*)
      OPT_CORES="${i#*=}"
      shift # past argument=value
      ;;
    -s=*|--searchpath=*)
      SEARCHPATH="${i#*=}"
      shift # past argument=value
      ;;
    -l=*|--lib=*)
      LIBPATH="${i#*=}"
      shift # past argument=value
      ;;
    --default)
      DEFAULT=YES
      shift # past argument with no value
      ;;
    --verbose)
      OPT_VERBOSE=1
      shift
      ;; 
      --help)
       usage;
       shift
      ;;
    --skipcmake)
      OPT_SKIPCMAKE=1
      shift
      ;;
    --clear)
      OPT_CLEAR=1
      shift
      ;;
    --debug)
      OPT_DEBUG=1
      shift
      ;;
    --release)
      OPT_RELEASE=1
      ;; 
    *)
      # unknown option
      echo "Unknown options $i"
      exit
      ;;
  esac
done

if [ "$OPT_DEBUG" -eq 0 ]  && [ "$OPT_RELEASE" -eq 0 ]; then
 echo "ERROR REQUIRED: --debug --release"	
 usage;
fi

echo "OPT_VERBOSE = $OPT_VERBOSE";
echo "OPT_CORES = $OPT_CORES";

