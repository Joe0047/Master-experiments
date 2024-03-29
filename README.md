# 安裝 Gurobi 及實驗流程

## 1. 安裝 Gurobi
請按照以下[李家岩](https://management.ntu.edu.tw/IM/faculty/teacher/sn/388)教授的教學網址，申請 Gurobi 帳號以及 Gurobi license，並且安裝 Gurobi 和 Anaconda。

* 若為 Windows，請參考:\
https://github.com/wurmen/Gurobi-Python/blob/master/Installation/%E5%AE%89%E8%A3%9D%E6%95%99%E5%AD%B8.md

* 若為 Linux，請參考:\
https://github.com/PO-LAB/Python-Gurobi-Pulp/blob/master/Installation/installation-for-linux.md

## 2. 執行實驗
請下載 [Master-experiments](https://github.com/Joe0047/Master-experiments)，並打開 [Master-experiments/Experiments/coflowSim/](https://github.com/Joe0047/Master-experiments/tree/main/Experiments/coflowSim)，以下為可執行的主程式 .py 檔:

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

若要使用 Spyder 或 Jupyter 等其他程式編譯器，請確認是否有安裝 gurobi package，之後打開 xxx.py，按下執行鍵便可執行；若要使用 cmd，請至主程式 .py 所在的位置，打開 cmd 並輸入 python xxx.py，便可執行。

