set -e
set -x
touch read.cpp
mv read.cpp read-$(date +"%Y-%m-%d_%H-%M-%S").cpp

hipcc 256_256_64_32_with16x32.cpp -E -C -DKITTENS_CDNA4 --offload-arch=gfx950 -std=c++20 -w --save-temps -L/usr/lib/python3.12/config-3.12-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu  -ldl  -lm  -I/A/HipKittens/include -I/A/HipKittens/prototype -I/usr/include/python3.12 -I/home/tazhu/.local/lib/python3.12/site-packages/pybind11/include -shared -fPIC -Rpass-analysis=kernel-resource-usage -I/A/HipKittens/include -I/opt/rocm/include/hip -I/usr/local/lib/python3.12/dist-packages/pybind11/include > read2.cpp

# /opt/rocm-7.0.0/bin/hipcc 256_256_64_16.cpp -E -C -DKITTENS_CDNA3 --offload-arch=gfx942 -std=c++20 -w --save-temps -I/opt/venv/lib/python3.10/site-packages/pybind11/include -I/root/HipKittens//include -I/root/HipKittens//prototype -I/usr/include/python3.10 -I/usr/local/lib/python3.12/dist-packages/pybind11-3.0.1-py3.12.egg/pybind11/include -L/usr/lib/python3.12/config-3.12-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu  -ldl  -lm  -shared -fPIC -Rpass-analysis=kernel-resource-usage -I/root/HipKittens//include -I/opt/rocm-7.0.0/include -I/opt/rocm-7.0.0/include/hip > read2.cpp

python extract.py