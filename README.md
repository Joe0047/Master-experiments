# 安裝Gurobi及實驗流程

## 1. 安裝Gurobi
請按照以下[李家岩](https://management.ntu.edu.tw/IM/faculty/teacher/sn/388)教授的教學網址，申請Gurobi帳號以及Gurobi license，並且安裝Gurobi和Anaconda。

* 若為Windows，請參考:\
https://github.com/wurmen/Gurobi-Python/blob/master/Installation/%E5%AE%89%E8%A3%9D%E6%95%99%E5%AD%B8.md

* 若為Linux，請參考:\
https://github.com/PO-LAB/Python-Gurobi-Pulp/blob/master/Installation/installation-for-linux.md

## 2. 執行實驗
請下載 [Master-experiments](https://github.com/Joe0047/Master-experiments)，並打開 [Master-experiments/Experiments/coflowSim/](https://github.com/Joe0047/Master-experiments/tree/main/Experiments/coflowSim)，以下為可執行的主程式.py檔:

* main_benchmark_divisible.py
* main_benchmark_indivisible.py
* main_custom_divisible.py
* main_custom_divisible_CDF.py
* main_custom_divisible_box_plot.py
* main_custom_divisible_dense_and_combined.py
* main_custom_divisible_heterogeneous.py
* main_custom_divisible_num_of_core.py
* main_custom_divisible_time_complexity.py
* main_custom_indivisible.py
* main_custom_indivisible_CDF.py
* main_custom_indivisible_box_plot.py
* main_custom_indivisible_dense_and_combined.py
* main_custom_indivisible_num_of_core.py
* main_custom_indivisible_time_complexity.py

若要使用Spyder或Jupyter等其他程式編譯器，請確認是否有安裝gurobi package，之後打開xxx.py檔，按下執行鍵便可執行；若要使用cmd，請至主程式.py所在的位置，打開cmd並輸入python xxx.py，便可執行。

