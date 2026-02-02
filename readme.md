

这是 LuckyRL 用来测试的仓库，
主要包含两个部分

### 1 rollout 的生成

在 [szz_run.sh](./szz_run.sh) 里边调用 [szz_eval_gen.sh](./szz_eval_gen.sh) 来生成 所有的 rollout

生成结果会保存在 szz_eval_outputs folder 下边

### 2 grading answer

在 [szz_grade.sh](./szz_grade.sh) 会自动把 szz_eval_outputs 下边没有 grade.txt 的所有文件都 grade 一遍


### 3 collect answer

用这个脚本 [collect_mean.py](./collect_mean.py) 会把所有的 grade.txt 读出来，变成能够直接复制粘贴到 excel 里边的给是，放到 paste.txt 文件里边






