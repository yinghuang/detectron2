## Changed files list

对比官方code发生变化的文件列表

1. `detectron2\detectron2\layers\csrc\cocoeval\cocoeval.cpp` line 483, For build on win10

2. `detectron2\detectron2\solver\build.py` line 163, For add min learning rate of WarmupCosineLR

3. `detectron2\detectron2\solver\lr_scheduler.py` line 60, line 67, line 80, For add min learning rate of WarmupCosineLR

4. `detectron2\detectron2\config\defaults.py` line 514, For add min learning rate of WarmupCosineLR


## Summary

改动汇总

1. win10编译: 1

2. WarmupCosineLR添加最低学习率: 2, 3, 4