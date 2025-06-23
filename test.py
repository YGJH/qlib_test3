import markdown        # pip install markdown
import json

a='''
在你的 test_epoch 里，你先对模型输出做了

这一步把 pred 从 GPU Tensor 变成了 NumPy array，接着又把它和还留在 CUDA 上的 label（一个 Tensor）一起丢到你的 loss_fn 里去计算。由于 NumPy 无法自动把一个在 GPU 上的 Tensor 转成它的 array，就在内部触发了 __array__()，进而报出了：

“TypeError: can’t convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.”

要解决它，思路就是 不要混合使用 NumPy 和 GPU Tensor 去算 loss/metric。你可以：

保持 pred、label 都是 Tensor，先在 GPU 上算完 loss/score（不要 .numpy()），再把最终的 .item() 或 .cpu().numpy() 转成 Python 数值。
或者如果真的要用 NumPy，把 label 也 .cpu().numpy()，保证二者都在主机内存中再算。
最推荐的做法就是像我们之前示例那样，把整个评估逻辑都留在 Tensor 里，结束后再 .item()／.cpu() 转回 Python 标量。这样既简单又不会跑出类型转换的错。

'''