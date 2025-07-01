

# main.py
from failure import FailurePredictionPipeline

if __name__ == '__main__':
    pipeline = FailurePredictionPipeline(file_name='full_devices.csv')
    pipeline.run()