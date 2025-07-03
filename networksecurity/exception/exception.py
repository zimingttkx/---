import sys
from networksecurity.logging import logger
class NetworkSecurityException(Exception):
    def __init__(self, error_message,error_details:sys):
        self.error_message = error_message # 错误信息
        self.error_details = error_details # 错误详情
        _,_,exc_tb = error_details.exc_info() # 完整的错误信息

        self.lineno = exc_tb.tb_lineno # 错误发生的行号
        self.filename = exc_tb.tb_frame.f_code.co_filename # 错误发生的文件名

    def __str__(self):
        return f" 错误发生在文件 {self.filename} 第 {self.lineno}行 : 内容是: {self.error_message}"



