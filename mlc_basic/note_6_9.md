# Note from 6-9
# Note6
比较有用的是 Torch -> TVM 的那部分的代码
- fx.symbolic_trace -> fx_graph -> mapper -> relax

```python
def map_param(param: nn.Parameter):
    ndim = len(param.data.shape)
    return relax.const(
        param.data.cpu().numpy(), relax.DynTensorType(ndim, "float32")
    )

def fetch_attr(fx_mod, target: str):
    """Helper function to fetch an attr"""
    target_atoms = target.split('.')
    attr_itr = fx_mod
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
        attr_itr = getattr(attr_itr, atom)
    return attr_itr

def from_fx(fx_mod, input_shapes, call_function_map, call_module_map):
    input_index = 0
    node_map = {}
    named_modules = dict(fx_mod.named_modules())

    bb = relax.BlockBuilder()

    fn_inputs = []
    fn_output = None
    with bb.function("main"):
        with bb.dataflow():
            for node in fx_mod.graph.nodes:
                if node.op == "placeholder":
                    # create input placeholder
                    shape = input_shapes[input_index]
                    input_index += 1
                    input_var = relax.Var(
                        node.target, shape, relax.DynTensorType(len(shape), "float32")
                    )
                    fn_inputs.append(input_var)
                    node_map[node] = input_var
                elif node.op == "get_attr":
                    node_map[node] = map_param(fetch_attr(fx_mod, node.target))
                elif node.op == "call_function":
                    node_map[node] = call_function_map[node.target](bb, node_map, node)
                elif node.op == "call_module":
                    named_module = named_modules[node.target]
                    node_map[node] = call_module_map[type(named_module)](bb, node_map, node, named_module)
                elif node.op == "output":
                    output = node_map[node.args[0]]
                    assert fn_output is None
                    fn_output = bb.emit_output(output)
        # output and finalize the function
        bb.emit_func_output(output, fn_inputs)
    return bb.get()
```


# Lecture 7
这个 Lecture 比较重要，主要介绍 

1. GPU 处理： single instruction multi-thread
   1. 一个程序属于一个单独线程，线程可以映射到 core
   2. 线程可以组合在一起形成一个线程块 threadblock，不同的 threadblock 可以映射到不同的流处理器上 （stream）
      1. threadidx.x 确定某个线程
   3. 多个 threadblock 的组合形成了一个 grid
      1. blockidx.x, blockidx.y 确定某个线程块

2. Hierachy 优化
   1. 利用存储结构：Shared Memory Blocking
    - 用 cache block 减少对 global mem 的读取
    - Cooperative Fetching：不同线程合作搬运数据

# Lecture8
特殊加速器
1. Scalar, Vector, Matrix Computation
2. 特殊内存 （special memory scopr）
   1. 特殊运算指令
   2. 拷贝指令： 全局到底部，底部到全局
3. Blockrize: 把单个 block 的计算做映射
   1. 目标是利用内存层级
4. Tensorized：讲 blockrize 的计算映射到张量单元
   1. 目的是利用硬件的核心

# Lecture 9
计算图优化
