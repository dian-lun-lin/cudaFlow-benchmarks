
TIMES=10
AVERAGE=0

echo "Test times: $TIMES"

##SerialGraph
#num_nodes_vec=(1024 4096 16384 65536)
#echo "==========Linear chain=========="
#for num_nodes in ${num_nodes_vec[@]}; do
  #echo "Number of nodes: $num_nodes"
  #for ((i=1; i<=TIMES; ++i)); do
    #echo "Iter $i:"
    #OUTPUT=$(./simple_graph -g SerialGraph --args $num_nodes)
    #AVERAGE=$((AVERAGE + OUTPUT))
    #echo "$OUTPUT ms"
  #done
  #AVERAGE=$((AVERAGE / TIMES))
  #echo "AVERAGE: $AVERAGE ms"
  #AVERAGE=0
  #echo " "
#done

#echo " "


#ParallelGraph
num_nodes_vec=(1024 4096 16384)
echo "==========Embarassing parallelism=========="
for num_nodes in ${num_nodes_vec[@]}; do
  echo "Number of nodes: $num_nodes"
  for ((i=1; i<=TIMES; ++i)); do
    echo "Iter $i:"
    OUTPUT=$(./simple_graph -g ParallelGraph --args $num_nodes)
    AVERAGE=$((AVERAGE + OUTPUT))
    echo "$OUTPUT ms"
  done
  AVERAGE=$((AVERAGE / TIMES))
  echo "AVERAGE: $AVERAGE ms"
  AVERAGE=0
  echo " "
done
echo " "

AVERAGE=0

#Tree
echo "==========Binary tree=========="
num_levels=(4 8 10 12 14)
for l in ${num_levels[@]}; do
  echo "Layers: $l"
  for ((i=1; i<=TIMES; ++i)); do
    echo "Iter $i:"
    ./simple_graph -g Tree --args 2 $l
    OUTPUT=$(./simple_graph -g Tree --args 2 $l)
    AVERAGE=$((AVERAGE + OUTPUT))
    echo "$OUTPUT ms"
  done
  AVERAGE=$((AVERAGE / TIMES))
  echo "AVERAGE: $AVERAGE ms"
  AVERAGE=0
  echo " "
done
echo " "

#Map Reduce
echo "==========Map reduce=========="
length=(128 256 512 1024)
for lg in ${length[@]}; do
  echo "Length: $lg"
  for ((i=1; i<=TIMES; ++i)); do
    echo "Iter $i:"
    OUTPUT=$(./simple_graph -g Diamond --args 16 $lg)
    AVERAGE=$((AVERAGE + OUTPUT))
    echo "$OUTPUT ms"
  done
  AVERAGE=$((AVERAGE / TIMES))
  echo "AVERAGE: $AVERAGE ms"
  AVERAGE=0
  echo " "
done
echo " "

#RandomDAG
echo "==========RandomDAG=========="
length=(128 256 512 1024)
for lg in ${length[@]}; do
  echo "Length: $lg"
  for ((i=1; i<=TIMES; ++i)); do
    echo "Iter $i:"
    OUTPUT=$(./simple_graph -g RandomDAG --args $lg 50 5)
    AVERAGE=$((AVERAGE + OUTPUT))
    echo "$OUTPUT ms"
  done
  AVERAGE=$((AVERAGE / TIMES))
  echo "AVERAGE: $AVERAGE ms"
  AVERAGE=0
  echo " "
done
echo " "
