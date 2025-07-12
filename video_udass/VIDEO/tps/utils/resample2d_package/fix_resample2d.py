import re

input_path = 'resample2d_kernel.cu'
output_path = 'resample2d_kernel_fixed.cu'

with open(input_path, 'r') as f:
    code = f.read()

# 替换 Tensor.data<T>() 为 data_ptr<T>()
code = re.sub(r'\.data<(\w+)>', r'.data_ptr<\1>', code)

# 修复 float * half 模式，替换为 __half2float 显式转换
code = re.sub(r'(\w+)\s*=\s*([0-9.]+f?)\s*\*\s*(\w+)\s*;', r'\1 = \2 * __half2float(\3);', code)

# 添加 cuda_fp16 头文件（如果没有）
if '#include <cuda_fp16.h>' not in code:
    code = '#include <cuda_fp16.h>\n' + code

with open(output_path, 'w') as f:
    f.write(code)

print(f"✅ 修复完成，已写入：{output_path}")



# cd ./VIDEO/tps/utils/resample2d_package
# python fix_resample2d.py
# mv resample2d_kernel_fixed.cu resample2d_kernel.cu
# python setup.py clean
# python setup.py build
# python setup.py install