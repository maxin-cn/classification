import os
import sys
from refiner import refine

onnx_model = sys.argv[1]
cache_path = sys.argv[2]
range_path = sys.argv[3]

refine(onnx_model, cache_path, range_path)
