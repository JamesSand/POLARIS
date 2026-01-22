

grading 的 code 在这里

evaluation/grade.py

和原始的 polaris 相比只修复了两个地方
1 原始的 grading 只支持 30 question 的 eval（只支持 aime 的 eval），我扩展到了支持任意数量的 question
2 原始的 grading 里边只支持一个 gt，现在改成了支持一个 list 传进来的 gt


这个 minerva 和 olympiad ground truth 是 list

这个 list 还不能 json serialization

所以现在的解法是，存在 parquet 里边的时候用 str 的格式存，

在 eval score 的时候，把这个东西转换回 list，只要 math 这个 list 里边的一个答案，就算对




