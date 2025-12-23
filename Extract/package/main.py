# package/main.py
import sys
import os
from pipeline.pipeline import DefaceCombinedPipeline
from package.config.config import PipelineConfig

# Thêm đường dẫn vào sys.path để Python có thể tìm thấy package của bạn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'package')))

def main():
    config = PipelineConfig()
    pipeline = DefaceCombinedPipeline(config)
    pipeline.run(start_index=0)  # Bạn có thể thay đổi start_index ở đây nếu muốn

if __name__ == "__main__":
    main()
