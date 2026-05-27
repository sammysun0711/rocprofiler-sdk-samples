# RCCL trace example

This section demonstrates how to trace RCCL using rocprofv3 across multiple GPUs.

### Steps

1. Verfiy rccl installed in the env
```
ls /opt/rocm/lib | grep rccl
```

2. Build [rccl-tests](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-systems/examples/rccl/rccl-tests) 


```
export PATH=$PATH:/opt/rocm/bin
cd rcc-tests
make clean && make
```

3. Run basic tests

for example
```
./all_reduce_perf -b 1G -e 1G -f 2 -g 8 -n 1 -w 0
``` 
- b	1G	Begin message size = 1 GB
- e	1G	End message size = 1 GB
- f	2	Size multiplication factor between iterations (no effect here since begin == end)
- g	8	Use 8 GPUs in a single process (single-node multi-GPU)
- n	1	Run 1 measured iteration
- w	0	No warmup iterations

you can regulate the command line.

4. Run profile

```bash
 rocprofv3 \
  --rccl-trace \
  --hip-trace \
  --sys-trace \
  --output-format pftrace \
  -- ./all_reduce_perf -b 1G -e 1G -f 2 -g 8 -n 1 -w 0
```

---

## References

- [Application tracing and profiling using rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocprofv3.html#rccl-trace)
- [rocprofiler-systems examples](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-systems/examples/rccl)
---

## License

This demonstration code is provided for educational purposes.

## Author

Created for demonstrating GPU kernel optimization techniques on AMD MI350 series GPUs.