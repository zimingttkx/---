"""

这个文件定义了一个自定义异常类 NetworkSecurityException，
该类继承自 Python 内置的 Exception 类，用于处理网络安全相关的异常。
当在网络安全相关的代码中发生异常时，可以抛出这个自定义异常，以便更好地记录和调试异常信息。
"""
import sys

class NetworkSecurityException(Exception):
    def __init__(self, error_message: Exception, error_detail: sys): # 参数名建议统一
        """
        用一个详细的、格式化的消息来初始化父类 Exception
        """
        super().__init__(f"错误发生在文件 {error_detail.exc_info()[2].tb_frame.f_code.co_filename} "
                         f"第 {error_detail.exc_info()[2].tb_lineno}行, "
                         f"内容是: {error_message}")
    def __str__(self):
        return self.args[0]


