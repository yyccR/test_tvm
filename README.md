# Test TVM in Macos

## 交叉编译android动态库`libtvm_runtime.so`
```
git clone --recursive https://github.com/apache/tvm tvm

# arm64-v8a
mkdir build_arm64 && cd build_arm64
cmake ../ \
      -DCMAKE_TOOLCHAIN_FILE=/path_to/Android/sdk/ndk/21.4.7075529/build/cmake/android.toolchain.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DANDROID_ABI="arm64-v8a" \
      -DANDROID_STL=c++_static \
      -DCMAKE_BUILD_TYPE=Release \
      -DANDROID_NATIVE_API_LEVEL=android-30  \
      -DANDROID_TOOLCHAIN=clang++
make runtime -j8

# armeabi-v7a
mkdir build_armeabi && cd build_armeabi
cmake ../ \
      -DCMAKE_TOOLCHAIN_FILE=/path_to/Android/sdk/ndk/21.4.7075529/build/cmake/android.toolchain.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DANDROID_ABI="armeabi-v7a" \
      -DANDROID_STL=c++_static \
      -DCMAKE_BUILD_TYPE=Release \
      -DANDROID_NATIVE_API_LEVEL=android-30  \
      -DANDROID_TOOLCHAIN=clang++
make runtime -j8
```

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

![检测效果](/images/det_image.jpg)