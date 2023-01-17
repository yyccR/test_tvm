# Test TVM in Macos



## Macos编译`libtvm_runtime.dylib`动态库
```
https://tvm.apache.org/docs/install/from_source.html

git clone --recursive https://github.com/apache/tvm tvm
cd tvm && mkdir build && cp ../cmake/config.cmake ./

# 需要下载 llvm https://github.com/llvm/llvm-project/releases/download/llvmorg-15.0.7/clang+llvm-15.0.7-x86_64-apple-darwin21.0.tar.xz
# 解压后将 llvm目录/bin/llvm-config 链接到 tvm
# 修改 build/config.cmake 替换 set(USE_LLVM llvm目录/bin/llvm-config)

cmake ..
make -j8

```

## 检测效果

![检测效果]()