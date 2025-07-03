"""

这个文件定义了一个自定义异常类 NetworkSecurityException，
该类继承自 Python 内置的 Exception 类，用于处理网络安全相关的异常。
当在网络安全相关的代码中发生异常时，可以抛出这个自定义异常，以便更好地记录和调试异常信息。
"""
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



