#!/bin/bash

window_list=(2 3 4 5 6 7 8)
city_list=('A')
city_list2=('A' 'B' 'C' 'D' 'E')



for city in "${city_list[@]}"

do

  for window in "${window_list[@]}"; do

  	data="--data ./data/data/SIGIR5/data/city_${city}.txt"

	mat="--sim_mat ./data/data/SIGIR5/matrix/neigh_matrix_voronoi_${city}.txt"

	save_name="--save_name cnnrnn_res.hhs.w-${window}.h-1.ratio.0.01.hw-4.pt"

	city_name="city_${city}"

	cmd="python main.py --normalize 2 --epochs 200 ${data} ${mat} --model CNNRNN_Res --dropout 0.2 --ratio 0.01 --residual_window 4 --save_dir mysave ${save_name} --horizon 1 --window ${window} --gpu 0 --metric 0 --city_name ${city_name} "

	echo $cmd

	eval $cmd

  done

done
