

真的是服了hhh

这个 minerva 和 olympiad ground truth 是 list

这个 list 还不能 json serialization

所以现在的解法是，存在 parquet 里边的时候用 str 的格式存，

在 eval score 的时候，把这个东西转换回 list，只要 math 这个 list 里边的一个答案，就算对




