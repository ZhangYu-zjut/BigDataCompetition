# Big Data Competition
First of all, I would like to thank to my teammates(**Xiao xuan Cui, Lin Yao, Jing ying Nong**). Without their help and support, I can't go so far!

For more information and details about data processing, please visit [https://github.com/calidentor/BigDataCompetition_YL](https://github.com/calidentor/BigDataCompetition_YL
)


## Introduction
This code is built on the basis of another code, which is used for epidemic prediction.(SIGIR 18)
For original version of code, please visit [https://github.com/CrickWu/DL4Epi](https://github.com/CrickWu/DL4Epi)

Our team modified the original code to adjust to Big Data Competition.

## Original Paper
[Deep Learning for Epidemiological Predictions](https://raw.githubusercontent.com/CrickWu/crickwu.github.io/master/papers/sigir2018.pdf) - Yuexin Wu et. al. SIGIR 2018


## Dependencies.
`Python == 2.7, Pytorch == 1.0.0`

## Quick start
### 1. For the use of **original data** (From US epidemic data)(**Not useful for competition**)

```
python main.py --normalize 1 --epochs 2 --data ./data/us_hhs/data.txt --sim_mat ./data/us_hhs/ind_mat.txt --model CNNRNN_Res \
--dropout 0.5 --ratio 0.01 --residual_window 4 --save_dir save --save_name cnnrnn_res.hhs.w-16.h-1.ratio.0.01.hw-4.pt \
--horizon 1 --window 16 --gpu 0 --metric 0
```

### 2. For the use of **Big Data Competition data** (From Baidu infection data)

#### 2.1 Train cities all together(recommend)

If you want to train **all cities**, just modify three parameters: `--data`, `--sim_mat`, `--city_name`
```
python main.py --normalize 2 --epochs 200 --data ./data/data/SIGIR/data.txt --sim_mat ./data/data/SIGIR5/matrix/neigh_matrix_voronoi_all.txt --model CNNRNN_Res --dropout 0.2 --ratio 0.01 --residual_window 4 --save_dir mysave  --save_name cnnrnn_res.hhs.w-16.h-1.ratio.0.01.hw-4.pt  --horizon 1 --window 6 --gpu 3 --metric 0 --city_name city_all
```

#### 2.2 Train cities separately(Not recommend)

An example of training **City_A**

```
python main.py --normalize 2 --epochs 60 --data ./data/data/SIGIR5/data/city_A.txt --sim_mat ./data/data/SIGIR5/matrix/neigh_matrix_voronoi_A.txt --model CNNRNN_Res --dropout 0.2 --ratio 0.01 --residual_window 4 --save_dir mysave  --save_name cnnrnn_res.hhs.w-16.h-1.ratio.0.01.hw-4.pt  --horizon 1 --window 6 --gpu 3 --metric 0 --city_name city_A
```

If you want to train **city_B**, just modify three parameters: `--data`, `--sim_mat`, `--city_name`
```
python main.py --normalize 2 --epochs 200 --data ./data/data/SIGIR5/data/city_B.txt --sim_mat ./data/data/SIGIR5/matrix/neigh_matrix_voronoi_B.txt --model CNNRNN_Res --dropout 0.2 --ratio 0.01 --residual_window 4 --save_dir mysave  --save_name cnnrnn_res.hhs.w-16.h-1.ratio.0.01.hw-4.pt  --horizon 1 --window 6 --gpu 3 --metric 0 --city_name city_B
```

If you want to train **city_C**, just modify three parameters: `--data`, `--sim_mat`, `--city_name`
```
python main.py --normalize 2 --epochs 200 --data ./data/data/SIGIR5/data/city_C.txt --sim_mat ./data/data/SIGIR5/matrix/neigh_matrix_voronoi_C.txt --model CNNRNN_Res --dropout 0.2 --ratio 0.01 --residual_window 4 --save_dir mysave  --save_name cnnrnn_res.hhs.w-16.h-1.ratio.0.01.hw-4.pt  --horizon 1 --window 6 --gpu 3 --metric 0 --city_name city_C
```

If you want to train **city_D**, just modify three parameters: `--data`, `--sim_mat`, `--city_name`
```
python main.py --normalize 2 --epochs 200 --data ./data/data/SIGIR5/data/city_D.txt --sim_mat ./data/data/SIGIR5/matrix/neigh_matrix_voronoi_D.txt --model CNNRNN_Res --dropout 0.2 --ratio 0.01 --residual_window 4 --save_dir mysave  --save_name cnnrnn_res.hhs.w-16.h-1.ratio.0.01.hw-4.pt  --horizon 1 --window 6 --gpu 3 --metric 0 --city_name city_D
```

If you want to train **city_E**, just modify three parameters: `--data`, `--sim_mat`, `--city_name`
```
python main.py --normalize 2 --epochs 200 --data ./data/data/SIGIR5/data/city_E.txt --sim_mat ./data/data/SIGIR5/matrix/neigh_matrix_voronoi_E.txt --model CNNRNN_Res --dropout 0.2 --ratio 0.01 --residual_window 4 --save_dir mysave  --save_name cnnrnn_res.hhs.w-16.h-1.ratio.0.01.hw-4.pt  --horizon 1 --window 6 --gpu 3 --metric 0 --city_name city_E
```

## Parameters Description
For `main.py`

```
output_print: whether print the model output during the train peocess.
	0: Don't print the output of intermediate process.
	1: Print the output of intermediate process.
```

```
result_vis: whether display the plot curve of predict result(using matplotlib).
	0: Don't display the visualization of result.
	1: Display the visualization of result.
```

```
output_print: normalization options
	0: no normalization
	1: global/matrix-wise normalization
	2: signal/column-wise normalization (Original code says **row-wise normalization**, which i think is incorrect !!)
```


## Option Explanation
For `main.py`

```
normalize: normalization options
	0: no normalization
	1: global/matrix-wise normalization
	2: signal/column-wise normalization 
	(Original code says **row-wise normalization**, which i think is incorrect !!)
```	

## Log Format

rse: Root-mean-square error

rae: Absolute error

correlation: Pearson correlation score

More information can be found in `main.py --help`.


## Version and modify history
-----------------origin version-----------------


**----------------------------------v1.0--------------------------------**

date: 2020.5.20

coder: Yu Zhang

code modify description:

In use:
1. (CNNRNN_Res.py -- line 16) `torch.Tensor` change to `torch.ones`
2. (utils.py -- line 102) add parameter `offset`, which can change the window start index
3. (command line) add parameter `city_name`, which means the city name you want to train and save.
4. (command line) add parameter `start_index`, which can change the window start index(corresponding to 'offset')
5. (main.py) add the function of saving results to the file.

**Not** in use:
1. (utils.py -- line 19) add `epsilon` (if x==0, log10(x) will still work)
2. (utils.py -- line 28) add log10 normalization.
3. (main.py -- line 104) add parameter `log_pre`，means whether do the log10 preproces. default is True，if you don't need, just add `--log_pre False` in the command line.


**----------------------------------v1.1----------------------------------**

date: 2020.5.24

coder: Yu Zhang

code modify description:

In use:
1. (file path: ./data/) add the file `predict.py`, which implements the function of predict future data. （used in main.py -- linne 205）
2. (file path: ./data/) add the file `pre_utils.py`, which implements the function of future data loader.（used in main.py -- linne 206）
3. (predict.py -- line 34) modify the save file name，add paraneter 'start_index', which means the start inedx of window.
4. (CNNRNN.sh -- line 132) add 'start_index_list'

**if you modified the code, you can write the details in the fllowing section**

**----------------------------------v1.2----------------------------------**

date: 2020.6.12

coder: Yu Zhang

code modify description:
1. (prediction.py) Review some bugs of original code.
	(1) NaN bug -- if denominator equals to zero, add a samll value(epsilon) to avoid division by zero.
	(2) index bug -- new_data[i] = old_data[i-1]
	(3) window size bug -- rollback the window size(p)
2. (main.py) Add two parameters in `mian.py`:'result_vis' and 'output_print'(has mentioned above).
3. (main.py) Add some codes to the whole project, including `data format transform` and `Visualization of prediction` (By Lin Yao)


**if you modified the code, you can write the details in the fllowing section**

**----------------------------------v1.3----------------------------------**

date: 2020.X.X

coder: XX

code modify description:

**Attention:**

Prediction result output path：./data/output/*.csv

