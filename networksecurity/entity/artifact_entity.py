""""
dataclass的作用是自动生成__init__和其他方法
比如第一个类如果不使用装饰器的话应该是:
class DataIngestionArtifactManual:
    def __init__(self, train_file_path: str, test_file_path: str):
        # 自动生成的构造函数
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path

    def __repr__(self) -> str:
        # 自动生成的可读性好的打印格式
        return f"DataIngestionArtifactManual(train_file_path='{self.train_file_path}', test_file_path='{self.test_file_path}')"

    def __eq__(self, other) -> bool:
        # 自动生成的相等比较
        if not isinstance(other, DataIngestionArtifactManual):
            return NotImplemented
        return (self.train_file_path == other.train_file_path and
                self.test_file_path == other.test_file_path)

"""
from dataclasses import dataclass

"""
数据获取阶段的产出物。当数据获取成功后，它会生成训练集和测试集文件。这个类就作为一个“包裹”，装着这两个文件的路径，然后传递给下一个阶段。
"""
@dataclass
class DataIngestionArtifact:
    train_file_path:str
    test_file_path:str

"""
数据验证阶段的产出物。这个阶段会检查数据的有效性。它完成之后，会生成一个包含更多信息的“包裹”，
比如：验证是否通过 (validation_status)、有效文件的路径、无效文件的路径、数据漂移报告的路径等
"""
@dataclass
class DataValidationArtifact:
    validation_status:bool
    valid_train_file_path:str
    valid_test_file_path:str
    invalid_train_file_path:str
    invalid_test_file_path:str
    drift_report_file_path:str


"""
数据转换阶段的产出物。当数据转换完成后，
它会生成转换后的训练/测试数据文件（.npy 格式）和一个预处理对象文件（.pkl 格式）。这个类就是装着这三个关键文件路径的“包裹”。
"""
@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path:str
    transformed_test_file_path:str