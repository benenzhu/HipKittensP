with open("read2.cpp", "r", encoding="utf-8") as f:
    lines = f.readlines()
import os
os.system("rm -rf read2.cpp")
os.system("make clean")
text = ""
cnt = 0
for line in lines:
    if "namespace ____start{" == line.strip():
        text = """
#include <memory>
#include <string>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <stdint.h>
#include <type_traits>
#include <concepts>
#include <memory>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <pybind11/pybind11.h>
#include <iostream>
namespace ____start{

"""
        print("find one")
        cnt += 1
    else: 
        if cnt == 2:
            text += line
with open("read.cpp", "w", encoding="utf-8") as f:
    f.write(text)
