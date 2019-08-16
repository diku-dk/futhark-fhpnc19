# Experiments for the paper *Compositional Deep Learning in Futhark*

There are two Futhark test programs: ``conv.fut`` and ``mlp.fut``.  To
run then, first construct their input data files by running ``make``.
Then, assuming you have installed the
[Futhark](https://futhark-lang.org) compiler and have a working OpenCL
setup, you can run the benchmarks:

```
futhark bench --backend=opencl conv.fut mlp.fut
```

Alternatively, to use the CUDA backend:

```
futhark bench --backend=cuda conv.fut mlp.fut
```

The two backends should perform roughly the same.  We used the OpenCL
backend for the paper.

These benchmarks were developed for Futhark 0.11.2.  If you read this
in the distant future, it is possible that the then-current version of
the compiler may not be able to compile them anymore.  If this
happens, please tell us.

## The library

The actual [deeplearning](https://github.com/HnimNart/deeplearning)
library is properly its own GitHub repository, but embedded here in
its entirety.  Run `futhark pkg upgrade && futhark pkg sync` if you
want to bump it to the newest version.
